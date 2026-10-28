#!/usr/bin/env python3
"""Validate the exact checkpoint selected by the FROSS Docker build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import fross_transformer as transformer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--required-model-steps", type=int, default=3000)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--vocab-size", type=int, default=5000)
    args = parser.parse_args()

    source_path = Path(args.model_source)
    if not source_path.is_file():
        raise SystemExit(f"Model source marker is missing: {source_path}")
    source = source_path.read_text(encoding="utf-8").strip()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Model checkpoint is missing: {checkpoint_path}")

    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise SystemExit(f"Could not read FROSS checkpoint: {type(exc).__name__}: {exc}") from exc

    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit("Checkpoint does not contain a metrics object.")
    checkpoint_version = checkpoint.get("version")
    # Brain 1.5 intentionally remains able to load the deployed 1.4 BPE checkpoint.
    # Accept that known legacy version while continuing to reject unknown versions;
    # tokenizer IDs, model shapes, vocabulary, and step count are checked separately.
    supported_checkpoint_versions = {transformer.VERSION, "FROSS Transformer 1.4"}
    if checkpoint_version not in supported_checkpoint_versions:
        raise SystemExit(
            f"Unsupported checkpoint version {checkpoint_version!r}; "
            f"supported versions: {', '.join(sorted(supported_checkpoint_versions))}."
        )
    if int(checkpoint.get("format_version", 0)) not in {1, 2}:
        raise SystemExit("Unsupported FROSS checkpoint format.")
    if int(metrics.get("vocab_size", 0)) != args.vocab_size:
        raise SystemExit(
            f"Vocabulary mismatch: expected {args.vocab_size}, got {metrics.get('vocab_size')}."
        )
    if metrics.get("vocab_target_reached") is not True:
        raise SystemExit("Checkpoint did not reach its requested vocabulary target.")
    if int(checkpoint.get("config", {}).get("block_size", 0)) != 320:
        raise SystemExit("Checkpoint context size does not match Brain 1.4 (320).")

    steps = int(metrics.get("training_steps", 0))
    if source == "uploaded-checkpoint":
        if steps < args.required_model_steps:
            raise SystemExit(
                f"Hosted checkpoint has {steps} training steps; "
                f"at least {args.required_model_steps} are required."
            )
    elif source.startswith("repository-training steps="):
        if steps != args.train_steps:
            raise SystemExit(
                f"Repository checkpoint has {steps} steps; expected {args.train_steps}."
            )
    else:
        raise SystemExit(f"Unknown checkpoint source marker: {source!r}")

    model, tokenizer, status = transformer.load_model(checkpoint_path)
    if model is None or tokenizer is None:
        raise SystemExit("Checkpoint failed model loading: " + json.dumps(status, ensure_ascii=False))
    if tokenizer.vocab_size != args.vocab_size:
        raise SystemExit(
            f"Loaded tokenizer has {tokenizer.vocab_size} tokens; expected {args.vocab_size}."
        )

    print(json.dumps({
        "ok": True,
        "modelSource": source,
        "version": checkpoint.get("version"),
        "training_steps": steps,
        "vocab_size": tokenizer.vocab_size,
        "parameters": metrics.get("parameters"),
        "checkpointLoaded": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
