#!/usr/bin/env python3
"""Train and compare FROSS checkpoints on a fixed held-out prompt set.

This is an experiment runner, not an automatic quality judge. Review the saved
answers against the criteria in the evaluation JSONL before choosing a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fross_transformer as transformer  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
        if not isinstance(row, dict) or not str(row.get("user", "")).strip():
            raise ValueError(f"Expected an object with non-empty 'user' at {path}:{line_no}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def run_variant(
    name: str,
    corpus_path: Path,
    eval_rows: list[dict[str, Any]],
    out_dir: Path,
    steps: int,
    batch_size: int,
    block_size: int,
    merges: int,
    seed: int,
) -> dict[str, Any]:
    checkpoint = out_dir / f"{name}.pt"
    print(f"\n=== {name}: {steps} training steps ===", flush=True)
    metrics = transformer.train_model(
        corpus_path,
        checkpoint,
        steps=steps,
        batch_size=batch_size,
        block_size=block_size,
        merge_count=merges,
        seed=seed,
    )
    model, tokenizer, status = transformer.load_model(checkpoint)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Could not reload {name}: {status}")
    outputs = []
    for item in eval_rows:
        prompt = str(item["user"]).strip()
        answer = transformer.generate_reply(
            model, tokenizer, prompt, max_new_tokens=120, mode="normal"
        )
        outputs.append({
            "user": prompt,
            "criteria": str(item.get("criteria", "")).strip(),
            "answer": answer,
        })
    result = {
        "variant": name,
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "outputs": outputs,
    }
    (out_dir / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FROSS training configurations.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/fross_corpus.jsonl")
    parser.add_argument("--eval-corpus", type=Path, default=ROOT / "data/fross_eval.jsonl")
    parser.add_argument(
        "--teacher-corpus", type=Path, default=None,
        help="Optional JSONL of reviewed teacher answers with the same user/assistant schema.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--baseline-steps", type=int, default=3000)
    parser.add_argument("--long-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=320)
    parser.add_argument("--merges", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.baseline_steps < 1 or args.long_steps < 1:
        parser.error("step counts must be positive")
    if not args.corpus.is_file() or not args.eval_corpus.is_file():
        parser.error("training corpus and evaluation corpus must both exist")

    base_rows = read_jsonl(args.corpus)
    eval_rows = read_jsonl(args.eval_corpus)
    if len(eval_rows) < 8:
        parser.error("evaluation corpus must contain at least 8 prompts")
    training_prompts = {str(row.get("user", "")).strip().casefold() for row in base_rows}
    eval_prompts = {str(row.get("user", "")).strip().casefold() for row in eval_rows}
    overlap = training_prompts & eval_prompts
    if overlap:
        parser.error(f"Evaluation prompts must be held out; found {len(overlap)} exact overlap(s)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    results.append(run_variant(
        "baseline", args.corpus, eval_rows, args.output_dir,
        args.baseline_steps, args.batch_size, args.block_size, args.merges, args.seed,
    ))
    results.append(run_variant(
        "more_steps", args.corpus, eval_rows, args.output_dir,
        args.long_steps, args.batch_size, args.block_size, args.merges, args.seed,
    ))

    if args.teacher_corpus is not None:
        if not args.teacher_corpus.is_file():
            parser.error(f"teacher corpus not found: {args.teacher_corpus}")
        teacher_rows = read_jsonl(args.teacher_corpus)
        if len(teacher_rows) < 8:
            parser.error("teacher corpus must contain at least 8 reviewed examples")
        # Keep eval prompts out of the augmented training set as well.
        teacher_overlap = eval_prompts & {
            str(row.get("user", "")).strip().casefold() for row in teacher_rows
        }
        if teacher_overlap:
            parser.error(f"Teacher corpus overlaps with {len(teacher_overlap)} evaluation prompt(s)")
        with tempfile.TemporaryDirectory(prefix="fross-distill-") as temp_dir:
            combined_path = Path(temp_dir) / "combined.jsonl"
            write_jsonl(combined_path, base_rows + teacher_rows)
            results.append(run_variant(
                "teacher_augmented", combined_path, eval_rows, args.output_dir,
                args.long_steps, args.batch_size, args.block_size, args.merges, args.seed,
            ))
    else:
        print(
            "\nTeacher variant skipped: pass --teacher-corpus with at least 8 "
            "reviewed teacher-generated JSONL examples.",
            flush=True,
        )

    summary = {
        "note": (
            "Validation loss values from different corpora are not directly comparable. "
            "Compare outputs on the same held-out prompts using the stated criteria; "
            "the teacher-augmented variant is supervised fine-tuning on teacher answers, "
            "not logit-level knowledge distillation."
        ),
        "variants": [
            {
                "variant": item["variant"],
                "checkpoint": item["checkpoint"],
                "train_loss": item["metrics"].get("train_loss"),
                "validation_loss": item["metrics"].get("validation_loss"),
                "training_steps": item["metrics"].get("training_steps"),
                "training_examples": item["metrics"].get("training_examples"),
                "validation_examples": item["metrics"].get("validation_examples"),
            }
            for item in results
        ],
        "manual_review_required": True,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved comparison to {args.output_dir / 'comparison.json'}")
    print("Review the same prompts side-by-side; do not select by training loss alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
