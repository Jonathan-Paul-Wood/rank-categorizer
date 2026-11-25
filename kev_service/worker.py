"""Subprocess entrypoints. Only this module imports Kev/torch."""
import json
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path


def checkpoint(path):
    from kev.checkpoint import Checkpoint
    return Checkpoint(path)


def validate(path, dataset):
    from kev.data import load_records, materialize
    from kev.model import MAX_STATE, fits, load_tokenizer, training_context
    ck = checkpoint(path)
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    records = load_records(dataset)
    for index, record in enumerate(records, 1):
        if not fits(materialize(record), tok, **training_context(MAX_STATE)):
            raise ValueError(f"record {index} exceeds Kev training context; shorten state/questions")


def train(path, job_dir, device):
    import torch
    ck = checkpoint(path)
    meta = ck.meta
    if ck.full:
        raise ValueError("full-weight fine-tuning is not supported")
    options = json.loads((Path(job_dir) / "job.json").read_text())["options"]
    # These settings must match the released adapter, not Kev's from-scratch defaults.
    adapter = ck.adapter_config()
    # Infer target preset from the actual adapter config, using the upstream model's module lists.
    # Standard checkpoints record the original training arguments in training_config.json when supplied.
    config_path = Path(path) / "training_config.json"
    original = json.loads(config_path.read_text()).get("args", {}) if config_path.exists() else {}
    targets = original.get("lora_targets", "all")
    # Released catalog checkpoints use all modules; reject a changed upstream adapter rather than partial-load it.
    if adapter.get("r") != meta.lora:
        raise ValueError("adapter rank and checkpoint metadata disagree")
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    args = ["kev.train", "--data", str(Path(job_dir) / "train.jsonl"),
            "--init_from", path, "--base", meta.base, "--base_revision", meta.base_revision or "",
            "--lora", str(meta.lora), "--head_dim", str(meta.head_dim),
            "--option_isolation", str(int(meta.option_isolation)),
            "--special_embeddings", str(int(meta.special_embeddings)), "--lora_targets", targets,
            "--weights_dtype", meta.weights_dtype, "--dtype", "bf16" if use_bf16 else "fp32",
            "--epochs", str(options["epochs"]), "--lr", str(options["learning_rate"]),
            "--batch", str(options["batch_size"]), "--accum", str(options["accumulation"]),
            "--seed", str(options["seed"]), "--checkpointing", "1", "--device", device,
            "--p_none", "0", "--p_distract", "0", "--p_none_distract", "0",
            "--out", str(Path(job_dir) / "staging" / "checkpoint")]
    sys.argv = args
    runpy.run_module("kev.train", run_name="__main__")
    from kev.checkpoint import read_meta, write_meta
    out = Path(job_dir) / "staging" / "checkpoint"
    updated = read_meta(out)
    updated.temperature = 1.0
    write_meta(out, updated)


def serve(path, device):
    import torch
    import uvicorn
    from kev.checkpoint import LoadOptions
    from kev.device import select
    from kev.serve import Server, app
    ck = checkpoint(path)
    opts = replace(LoadOptions.from_env(), backend="torch", cuda_graphs=False, fused=False)
    if opts.dtype is None:
        opts = replace(opts, dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
                       else torch.float32)
    device = select(device, dtype=opts.dtype or torch.float32)
    tok, model = ck.load(device, opts)
    app.state.server = Server(ck, tok, model, device)
    uvicorn.run(app, host="127.0.0.1", port=8010, access_log=False)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    operation, *args = sys.argv[1:]
    if operation == "serve":
        serve(*args)
    elif operation == "validate":
        validate(*args)
    elif operation == "train":
        train(*args)
    elif operation == "evaluate":
        path, dataset, out, device = args
        if os.environ.get("KEV_DTYPE") == "auto":
            import torch
            os.environ["KEV_DTYPE"] = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else "fp32"
        sys.argv = ["kev.benchmark", "--run", path, "--data", dataset, "--out", out, "--device", device]
        runpy.run_module("kev.benchmark", run_name="__main__")
        report = json.loads((Path(out) / "report.json").read_text())
        if report["coverage"]["rejected_records"] or not report["coverage"]["evaluated_records"]:
            raise ValueError("evaluation did not cover the entire held-out partition")
    else:
        raise ValueError("unknown worker operation")


if __name__ == "__main__":
    main()
