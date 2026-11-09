import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate

evaluate.reference()
from common import UPSTREAM, digest, load, question_id
from run import base_model, config


def test_both_sizes_share_native_full_test_and_decoding(tmp_path):
    resolutions = []
    for size in evaluate.SIZES:
        model = base_model(size)
        assert model.name == load("models.json")[size]["model_id"].split("/")[-1]
        args = (size, model, tmp_path / size, tmp_path / "data", "http://127.0.0.1:8000/retrieve", 32)
        cfg = evaluate.resolved_config(*args)
        native = config(size, model, args[2], args[5], evaluate=True, data_dir=args[3],
                        search_url=args[4], eval_scope="full")
        assert cfg.actor_rollout_ref.rollout == native.actor_rollout_ref.rollout
        assert cfg.env == native.env
        assert cfg.data.val_files == native.data.val_files == str(args[3] / "test.parquet")
        assert cfg.data.train_files == cfg.data.val_files
        assert cfg.trainer.val_only and cfg.trainer.resume_mode == "disable" and cfg.trainer.save_freq == -1
        assert cfg.trainer.logger == ["console"]
        assert cfg.trainer.rollout_data_dir is None and cfg.trainer.validation_data_dir is None
        assert cfg.trainer.log_val_generations == 0
        assert not cfg.actor_rollout_ref.actor.use_kl_loss
        assert cfg.ray_init.num_gpus == cfg.trainer.n_gpus_per_node == 4
        env = cfg.ray_init.runtime_env.env_vars
        assert env.QA_EVAL_MODE == env.QA_INITIAL_EVAL_SCOPE == "full"
        assert "QA_SUBSET_RECEIPT" not in env
        assert "validation_subset" not in OmegaConf.to_yaml(cfg)
        resolutions.append((cfg.data, cfg.env, cfg.actor_rollout_ref.rollout))
    assert resolutions[0] == resolutions[1]


def native_validation(monkeypatch):
    """Exercise the pinned native method, without pretending to run CUDA."""
    class Proto:
        def __init__(self, batch, non_tensor_batch):
            self.batch = batch
            self.non_tensor_batch = non_tensor_batch

        @classmethod
        def from_single_dict(cls, values):
            return cls({k: v for k, v in values.items() if torch.is_tensor(v)},
                       {k: v for k, v in values.items() if not torch.is_tensor(v)})

        def repeat(self, repeat_times, interleave):
            assert repeat_times == 1 and interleave
            return self

        def pop(self, batch_keys, non_tensor_batch_keys):
            return Proto({k: self.batch.pop(k) for k in batch_keys},
                         {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys})

    source = UPSTREAM / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    validate = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_validate")
    namespace = {"np": np, "torch": torch, "DataProto": Proto}
    exec(compile(ast.Module(body=[validate], type_ignores=[]), str(source), "exec"), namespace)
    native = ModuleType("verl.trainer.ppo.ray_trainer")

    def no_train(*args, **kwargs):
        raise AssertionError("Native training fit must not run")

    native.RayPPOTrainer = type("LockedNative", (), {"_validate": namespace["_validate"], "fit": no_train})
    monkeypatch.setitem(sys.modules, native.__name__, native)
    spec = importlib.util.spec_from_file_location("step0_adapter_test", evaluate.RUNTIME / "grpo_eval.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter, Proto


@pytest.mark.parametrize("mutation", ["none", "reordered", "missing", "lost_row", "retrieval_error", "training"])
def test_real_native_validation_boundary(tmp_path, monkeypatch, mutation):
    adapter, proto = native_validation(monkeypatch)
    questions = [dict(question=f"Q{i}", ground_truth={"target": ["A"]},
                      data_source="nq" if i < 2 else "hotpotqa") for i in range(3)]
    ids = [question_id(row) for row in questions]
    receipt = tmp_path / "full_test_receipt.json"
    receipt.write_text(json.dumps({"rows": {
        "count": 3, "question_ids": ids, "source_counts": {"nq": 2, "hotpotqa": 1},
    }}), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    model = load("models.json")["3b"]
    (run / "binding.json").write_text(json.dumps({"model": model}), encoding="utf-8")
    monkeypatch.setenv("GRPO_RUN_DIR", str(run))
    monkeypatch.setenv("QA_DATA_RECEIPT", str(receipt))
    monkeypatch.setenv("QA_EVAL_MODE", "full")
    actual = list(questions)
    if mutation == "reordered":
        actual[0], actual[1] = actual[1], actual[0]
    elif mutation == "missing":
        actual.pop()
    batches = []
    for offset in range(0, len(actual), 2):
        items = actual[offset:offset + 2]
        count = len(items)
        batches.append({
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
            "position_ids": torch.zeros((count, 2), dtype=torch.long),
            "raw_prompt_ids": np.array([[1, 1]] * count, dtype=object),
            "data_source": np.array([x["data_source"] for x in items], dtype=object),
            "env_kwargs": np.array(items, dtype=object),
            "index": np.arange(offset, offset + count),
        })
    calls = []

    def step(actions):
        anchor = (
            '<information>{"result":"Search error: HTTP503"}</information>'
            if mutation == "retrieval_error" else
            '<information>{"result":"The phrase Search error: appears inside a passage."}</information>'
        )
        return {"anchor": [anchor]}, None, None, None

    def collect(gen_batch, envs, is_train, **kwargs):
        assert not is_train
        assert "index" not in gen_batch.non_tensor_batch
        envs.step(["<search>Q</search>"])
        items = gen_batch.non_tensor_batch.pop("env_kwargs")
        local = [0, 0, 1] if len(items) == 2 else [0]
        if mutation == "lost_row":
            local = [0]
        calls.append(len(items))
        return proto({"responses": torch.ones((len(local), 2), dtype=torch.long)}, {
            "index": np.array(local),
            "data_source": np.array([items[i]["data_source"] for i in local]),
            "episode_rewards": np.array([float(items[i]["question"] != "Q1") for i in local]),
            "traj_uid": np.array([items[i]["question"] for i in local]),
            "tool_callings": np.zeros(len(local)),
        })

    trainer = adapter.SearchTrainer()
    trainer.config = evaluate.resolved_config("3b", base_model("3b"), run, tmp_path,
                                              "http://127.0.0.1:8000/retrieve", 32)
    trainer.val_dataloader = batches
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
    trainer.actor_rollout_wg = object()
    trainer.val_envs = SimpleNamespace(step=step)
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0, decode=lambda *a, **kw: "<answer>A</answer>")
    trainer.val_reward_fn = lambda batch, **kwargs: {
        "reward_tensor": torch.tensor([[0., score] for score in batch.non_tensor_batch["episode_rewards"]])}
    trainer._maybe_log_val_generations = lambda **kwargs: None

    if mutation == "training":
        trainer.config.trainer.val_only = False
    if mutation != "none":
        with pytest.raises(RuntimeError, match="ordered|missed|retriever failed|read-only"):
            trainer.fit()
        assert not (run / "metrics.json").exists()
    else:
        trainer.fit()
        assert calls == [2, 1]  # Must keep the last partial batch.
        assert {p.name for p in run.iterdir()} == {"binding.json", "metrics.json"}
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        assert result["update"] == 0 and result["scope"] == "full" and result["model"] == model
        assert result["ordered_question_ids_sha256"] == digest(ids)
        assert result["correct"] == 2 and result["questions"] == 3
        assert result["question_micro_em"] == pytest.approx(2 / 3)
        assert result["dataset_macro_em"] == pytest.approx(.75)
        with pytest.raises(RuntimeError, match="save checkpoints"):
            trainer._save_checkpoint()
        with pytest.raises(RuntimeError, match="load a trained checkpoint"):
            trainer._load_checkpoint()
    assert trainer.traj_collector.multi_turn_loop is collect
    assert trainer.val_envs.step is step


def test_result_binding_rejects_subset_and_wrong_model(tmp_path):
    counts = load("data.json")["splits"]["test"]["source_counts"]
    report = {
        "scope": "full", "update": 0, "questions": 51713, "correct": 0,
        "status": "completed", "model": load("models.json")["1.5b"],
        "ordered_question_ids_sha256": evaluate.FULL_ID_SHA,
        "by_dataset": {key: {"questions": count, "correct": 0, "em": 0.} for key, count in counts.items()},
        "question_micro_em": 0., "dataset_macro_em": 0.,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert evaluate.verify_summary(path, "1.5b") == report
    with pytest.raises(RuntimeError, match="pinned base model"):
        evaluate.verify_summary(path, "3b")
    report["scope"] = "subset"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="full-test result"):
        evaluate.verify_summary(path, "1.5b")
