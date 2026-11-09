"""Evaluate the pinned Qwen2.5 Instruct bases on native full Search-QA."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / ".deps" / "searchqa-grpo"
RUNTIME = ROOT / "runtime"
REPOSITORY = "https://github.com/leiye302/grpo-searchqa-qwen25-3b.git"
REVISION = "a054b5c69afd2d74b258dc8101682a6fbe2f04d7"
SIZES = ("1.5b", "3b")
FULL_QUESTIONS = 51713
FULL_ID_SHA = "b1c79fb8529c7ddbd1051e008995676baa5bb9dbe3c6d34a1ade2c0d2f14d427"
DEFAULT_DATA = ROOT / "data" / "searchqa"
DEFAULT_RETRIEVAL = ROOT / "data" / "retrieval"


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-C", str(directory), *args],
        text=True,
    ).strip()


def reference(fetch=False):
    if not REFERENCE.exists():
        if not fetch:
            raise RuntimeError("Run python evaluate.py setup first.")
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fetch-", dir=REFERENCE.parent) as temporary:
            staging = Path(temporary) / "checkout"
            staging.mkdir()
            git(staging, "init")
            git(staging, "remote", "add", "origin", REPOSITORY)
            git(staging, "fetch", "--depth", "1", "origin", REVISION)
            git(staging, "checkout", "--detach", REVISION)
            staging.rename(REFERENCE)
    if git(REFERENCE, "rev-parse", "HEAD") != REVISION or git(REFERENCE, "status", "--porcelain"):
        raise RuntimeError("Public Search-QA dependency is not the clean pinned revision.")
    source = str(REFERENCE / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_rows(rows):
    from common import load

    test = load("data.json")["splits"]["test"]
    if (rows["count"], rows["ordered_question_ids_sha256"], rows["source_counts"]) != (
        FULL_QUESTIONS, FULL_ID_SHA, test["source_counts"]
    ):
        raise RuntimeError("Prepared test questions, ordering or sources differ from the locked full set.")


def verify_test(directory):
    import pandas as pd
    from common import load, sha
    from data import row_summary

    receipt = read_json(directory / "full_test_receipt.json")
    if receipt["raw_sha256"] != load("data.json")["splits"]["test"]["sha256"]:
        raise RuntimeError("Raw test-file provenance differs.")
    path = directory / "test.parquet"
    if sha(path) != receipt["prepared_sha256"]:
        raise RuntimeError("Prepared test Parquet checksum differs.")
    rows = row_summary(pd.read_parquet(path, columns=["env_kwargs"]))
    check_rows(rows)
    if rows != receipt["rows"]:
        raise RuntimeError("Prepared test receipt differs from its actual question identities.")
    return receipt


def prepare_test(directory):
    import pandas as pd
    from common import atomic_json, load, sha
    from data import native_processor, row_summary
    from downloads import fetch

    if (directory / "full_test_receipt.json").exists():
        return verify_test(directory)
    target = directory / "test.parquet"
    if target.exists():
        raise RuntimeError("Uncertified test.parquet already exists; use a separate --data-dir.")
    directory.mkdir(parents=True, exist_ok=True)
    lock = load("data.json")
    raw = fetch(lock["repository"], lock["revision"], "test.parquet", directory / "raw", lock["splits"]["test"])
    process = native_processor()
    prepared = pd.read_parquet(raw).apply(lambda row: process(row, "test", row.name), axis=1)
    rows = row_summary(prepared)
    check_rows(rows)
    pending = directory / "test.parquet.pending"
    prepared.to_parquet(pending, index=False)
    pending.replace(target)
    atomic_json(directory / "full_test_receipt.json", {
        "raw_sha256": lock["splits"]["test"]["sha256"],
        "prepared_sha256": sha(target),
        "rows": rows,
    })
    return verify_test(directory)


def resolved_config(size, model, output, data_dir, search_url, cpus):
    from common import UPSTREAM
    from run import config

    cfg = config(size, model, output, cpus, evaluate=True, data_dir=data_dir,
                 search_url=search_url, eval_scope="full")
    # TaskRunner requires a nonempty training loader even when its loop is
    # never used. Use the existing test file as its inert input, not train data.
    cfg.data.train_files = cfg.data.val_files
    cfg.trainer.resume_mode = "disable"
    cfg.trainer.resume_from_path = None
    cfg.trainer.save_freq = -1
    cfg.trainer.test_freq = -1
    cfg.trainer.total_training_steps = 0
    cfg.trainer.log_val_generations = 0
    cfg.trainer.validation_data_dir = None
    cfg.trainer.rollout_data_dir = None
    # A training-only reference worker is not used for native validation.
    cfg.actor_rollout_ref.actor.use_kl_loss = False
    variables = cfg.ray_init.runtime_env.env_vars
    variables.PYTHONPATH = os.pathsep.join(map(str, (RUNTIME, REFERENCE / "src", UPSTREAM)))
    variables.QA_DATA_RECEIPT = str(data_dir / "full_test_receipt.json")
    del variables.QA_SUBSET_RECEIPT
    return cfg


def verify_summary(path, size):
    import math
    from common import load

    report = read_json(path)
    model = load("models.json")[size]
    if (report["scope"], report["update"], report["questions"], report["ordered_question_ids_sha256"]) != (
        "full", 0, FULL_QUESTIONS, FULL_ID_SHA
    ) or report["status"] != "completed" or report["model"] != model:
        raise RuntimeError("Not a complete update-0 full-test result for the pinned base model.")
    by_source = report["by_dataset"]
    if {key: row["questions"] for key, row in by_source.items()} != load("data.json")["splits"]["test"]["source_counts"]:
        raise RuntimeError("Full-test per-dataset counts differ.")
    for row in by_source.values():
        if (type(row["correct"]) is not int or not 0 <= row["correct"] <= row["questions"]
                or not math.isclose(row["em"], row["correct"] / row["questions"])):
            raise RuntimeError("Invalid per-dataset EM.")
    correct = sum(row["correct"] for row in by_source.values())
    if (report["correct"] != correct
            or not math.isclose(report["question_micro_em"], correct / FULL_QUESTIONS)
            or not math.isclose(report["dataset_macro_em"], sum(row["em"] for row in by_source.values()) / len(by_source))):
        raise RuntimeError("Invalid aggregate EM.")
    return report


def evaluate_one(args):
    """Isolated process: release Ray/vLLM before the next model starts."""
    from common import UPSTREAM, WORLD_SIZE, atomic_json, devices, load, sha
    from retrieval import verify as verify_retrieval
    from run import base_model, check_retriever, verify_model
    from setup import verify
    from omegaconf import OmegaConf

    if sys.platform != "linux":
        raise RuntimeError("Full evaluation requires the documented Linux/CUDA environment.")
    verify()
    verify_test(args.data_dir)
    verify_retrieval(args.retrieval_dir)
    model = base_model(args.size).resolve()
    # The pinned GRPO helper performs the base revision/SHA checks in this mode.
    verify_model(model, args.size, evaluate=False)
    check_retriever(args.search_url)
    cfg = resolved_config(args.size, model, args.output, args.data_dir, args.search_url, args.cpus)
    os.environ.update(dict(cfg.ray_init.runtime_env.env_vars),
                      CUDA_VISIBLE_DEVICES=",".join(devices(args.gpus)), RAY_ADDRESS="local")
    import torch

    if torch.cuda.device_count() != WORLD_SIZE or any(
        torch.cuda.get_device_properties(i).total_memory < 70 * 1024**3 for i in range(WORLD_SIZE)
    ):
        raise RuntimeError("Allocate four GPUs with at least 70 GiB visible memory each (recipe: A100 80GB).")
    atomic_json(args.output / "binding.json", {
        "model": load("models.json")[args.size],
        "model_receipt_sha256": sha(model / "download_receipt.json"),
        "reference_commit": REVISION,
        "sdar_commit": load("upstream.json")["commit"],
        "evaluation_protocol_sha256": sha(REFERENCE / "configs/evaluation.json"),
        "test_receipt_sha256": sha(args.data_dir / "full_test_receipt.json"),
        "retrieval_receipt_sha256": sha(args.retrieval_dir / "retrieval_receipt.json"),
        "ordered_question_ids_sha256": FULL_ID_SHA,
        "world_size": WORLD_SIZE,
    })
    sys.path[:0] = [str(RUNTIME), str(UPSTREAM)]
    os.chdir(UPSTREAM)
    import ray
    from verl.trainer.main_ppo import run_ppo

    with tempfile.TemporaryDirectory(prefix="qa0-") as temp:
        cfg.ray_init._temp_dir = temp
        OmegaConf.save(cfg, args.output / "resolved_config.yaml")
        try:
            run_ppo(cfg)
        finally:
            ray.shutdown()
    if {p.name for p in args.output.iterdir()} != {"run.log", "binding.json", "resolved_config.yaml", "metrics.json"}:
        raise RuntimeError("Unexpected output: expected only log, run metadata and metrics, no checkpoints or rollouts.")
    verify_summary(args.output / "metrics.json", args.size)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "prepare", "plan", "run", "_evaluate"))
    parser.add_argument("--size", choices=(*SIZES, "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cpus < 16:
        parser.error("Allocate at least 16 logical CPUs.")
    from urllib.parse import urlsplit

    url = urlsplit(args.search_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        parser.error("Use a credential-free native retriever URL, without a query string.")
    for name in ("data_dir", "retrieval_dir", "output"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    return args


def main():
    args = options()
    reference(fetch=args.command == "setup")
    from common import atomic_json, devices
    from run import base_model, verify_model
    from setup import download, framework, verify

    if args.command == "setup":
        framework()
        return
    verify()
    sizes = SIZES if args.size == "both" else (args.size,)
    if args.command == "prepare":
        receipt = prepare_test(args.data_dir)
        print(f"Verified the native full test: {receipt['rows']['count']:,} questions.")
        for size in sizes:
            model = base_model(size)
            if not (model / "download_receipt.json").exists():
                download(size)
            verify_model(model, size, evaluate=False)
        return
    devices(args.gpus)
    if args.command == "_evaluate":
        if len(sizes) != 1 or args.output is None:
            raise ValueError("Internal evaluation requires one size and a dedicated output directory.")
        evaluate_one(args)
        return
    if args.command == "plan":
        from omegaconf import OmegaConf

        for size in sizes:
            print(f"# {size}, update 0, full Search-QA ({FULL_QUESTIONS:,} questions)")
            print(OmegaConf.to_yaml(resolved_config(size, base_model(size), ROOT / "runs" / size,
                args.data_dir, args.search_url, args.cpus)))
        return
    if sys.platform != "linux":
        raise RuntimeError("Full evaluation requires the documented Linux/CUDA environment.")
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for size in sizes:
        run = output / size
        run.mkdir()
        log = run / "run.log"
        print(f"Running full Search-QA for {size}; follow {log}", flush=True)
        with log.open("x", encoding="utf-8") as stream:
            subprocess.run([
                sys.executable, "-u", str(Path(__file__).resolve()), "_evaluate",
                "--size", size, "--gpus", args.gpus, "--cpus", str(args.cpus), "--output", str(run),
                "--data-dir", str(args.data_dir), "--retrieval-dir", str(args.retrieval_dir),
                "--search-url", args.search_url,
            ], stdout=stream, stderr=subprocess.STDOUT, check=True)
        results[size] = verify_summary(run / "metrics.json", size)
        print(f"{size}: macro EM {results[size]['dataset_macro_em']:.2%}, "
              f"micro EM {results[size]['question_micro_em']:.2%}", flush=True)
    atomic_json(output / "summary.json", {"update": 0, "scope": "full", "results": results})
    print(f"Completed. {output / 'summary.json'}")


if __name__ == "__main__":
    main()
