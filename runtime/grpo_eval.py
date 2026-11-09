"""Metrics-only adapter loaded by the pinned SDAR TaskRunner."""

import json
import os
from pathlib import Path
import time

from common import atomic_json, load, plain, question_id
from evaluation_contract import summarize_questions
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def check_retrieval(anchors):
    # Detect only explicit native tool errors, not words inside retrieved passages.
    for index, observation in enumerate(anchors):
        if not isinstance(observation, str):
            continue
        text = observation.strip()
        start, end = "<information>", "</information>"
        if not (text.startswith(start) and text.endswith(end)):
            continue
        try:
            payload = json.loads(text[len(start):-len(end)])
        except ValueError:
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, str) and (
            result.startswith(("Search error:", "Error processing search results:", "Unknown API state ("))
            or result == "Search request failed or timed out after retries."
        ):
            raise RuntimeError(f"Native retriever failed at batch row {index}; full-test score not published.")


class SearchTrainer(RayPPOTrainer):
    def _load_checkpoint(self, *args, **kwargs):
        raise RuntimeError("Evaluation does not load a trained checkpoint.")

    def _save_checkpoint(self, *args, **kwargs):
        raise RuntimeError("Evaluation does not save checkpoints.")

    def fit(self):
        # Do not call the base fit: there is no path to an optimizer update here.
        cfg = self.config.trainer
        if (not cfg.val_only or cfg.resume_mode != "disable" or cfg.save_freq != -1
                or cfg.total_training_steps != 0
                or os.environ["QA_EVAL_MODE"] != "full" or self.val_reward_fn is None):
            raise RuntimeError("This entrypoint only supports full, read-only update-0 evaluation.")
        self.global_steps = 0
        self.val_reward_fn.num_examine = 0  # Disable upstream prompt/response samples in logs.
        self._validate_full()

    def _validate_full(self):
        run = Path(os.environ["GRPO_RUN_DIR"])
        expected = json.loads(Path(os.environ["QA_DATA_RECEIPT"]).read_text(encoding="utf-8"))["rows"]
        model = json.loads((run / "binding.json").read_text(encoding="utf-8"))["model"]
        started = time.monotonic()
        questions = []
        batches = 0
        original_collect = self.traj_collector.multi_turn_loop
        original_step = self.val_envs.step

        def guarded_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            check_retrieval(result[0]["anchor"])
            return result

        def capture(*args, **kwargs):
            nonlocal batches
            inputs = kwargs["gen_batch"].non_tensor_batch["env_kwargs"]
            identities = [question_id(row) for row in inputs]
            sources = [row["data_source"] for row in inputs]
            offset = len(questions)
            if (not identities or identities != expected["question_ids"][offset:offset + len(identities)]):
                raise RuntimeError("Actual validation inputs differ from the ordered full-test receipt.")
            batch = original_collect(*args, **kwargs)
            data = batch.non_tensor_batch
            episodes = {}
            for i in range(len(batch.batch["responses"])):
                local = int(data["index"][i])
                if not 0 <= local < len(identities) or str(data["data_source"][i]) != sources[local]:
                    raise RuntimeError("Native trajectory lost its input-question identity.")
                row = {
                    "index": offset + local, "question_id": identities[local], "data_source": sources[local],
                    "episode_reward": float(data["episode_rewards"][i]), "trajectory_id": str(data["traj_uid"][i]),
                }
                if local in episodes and episodes[local] != row:
                    raise RuntimeError("Conflicting episode results for a single question.")
                episodes[local] = row
            if len(episodes) != len(identities):
                raise RuntimeError("Native validation missed questions in a batch.")
            questions.extend(episodes.values())
            batches += 1
            print(f"Full evaluation: {len(questions):,}/{expected['count']:,} questions ({batches} batches)",
                  flush=True)
            return batch

        self.val_envs.step = guarded_step
        self.traj_collector.multi_turn_loop = capture
        try:
            native_metrics = super()._validate()
        finally:
            self.traj_collector.multi_turn_loop = original_collect
            self.val_envs.step = original_step
        if not native_metrics:
            raise RuntimeError("Native validation did not return metrics.")
        summary = summarize_questions(questions, expected["question_ids"], expected["source_counts"])
        summary.update(
            status="completed", scope="full", update=0, model=model, batches=batches,
            wall_seconds=time.monotonic() - started, native_metrics=plain(native_metrics),
        )
        atomic_json(run / "metrics.json", summary)
        print(f"Update 0: macro EM {summary['dataset_macro_em']:.2%}; "
              f"micro EM {summary['question_micro_em']:.2%}", flush=True)
