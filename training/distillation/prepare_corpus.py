#!/usr/bin/env python3
"""Validate and merge FROSS dialogue corpora while preserving teacher provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

def _text(value: Any, field: str, source: Path, line_number: int, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source}:{line_number}: {field} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{source}:{line_number}: {field} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{source}:{line_number}: {field} exceeds {maximum} characters")
    return value

def _normalise_record(row: Any, source: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{source}:{line_number}: each JSONL line must contain an object")
    record: dict[str, Any] = {
        "user": _text(row.get("user"), "user", source, line_number, 30000),
        "assistant": _text(row.get("assistant"), "assistant", source, line_number, 50000),
    }
    history = row.get("history", [])
    if history is None:
        history = []
    if not isinstance(history, list):
        raise ValueError(f"{source}:{line_number}: history must be a list")
    clean_history = []
    for index, turn in enumerate(history[-8:]):
        if not isinstance(turn, dict):
            raise ValueError(f"{source}:{line_number}: history item {index} must be an object")
        role = str(turn.get("role", "")).strip().lower()
        if role in {"assistant", "fross"}:
            role = "assistant"
        elif role == "user":
            role = "user"
        else:
            raise ValueError(f"{source}:{line_number}: history item {index} role must be user or assistant")
        value = turn.get("content")
        if not isinstance(value, str):
            raise ValueError(f"{source}:{line_number}: history item {index} content must be a string")
        value = " ".join(value.split()).strip()
        if value:
            clean_history.append({"role": role, "content": value[:700]})
    if clean_history:
        record["history"] = clean_history
    for key in ("source", "teacher_model"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            record[key] = value.strip()[:120]
    return record

def read_corpus(path: str | Path, required: bool = True) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Corpus file not found: {source}")
        return []
    rows = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            rows.append(_normalise_record(raw, source, line_number))
    return rows

def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    def norm(value: str) -> str:
        return " ".join(value.replace("ё", "е").lower().split())
    history = tuple((turn["role"], norm(turn["content"])) for turn in record.get("history", []))
    return history, norm(record["user"])

def merge_records(base_records: list[dict[str, Any]], teacher_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    merged = []
    seen = set()
    duplicates = 0
    for row in [*base_records, *teacher_records]:
        key = record_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        merged.append(row)
    if len(merged) < 8:
        raise ValueError(f"At least 8 unique examples are required; found {len(merged)}.")
    return merged, duplicates

def write_jsonl_atomic(path: str | Path, records: list[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return hashlib.sha256(target.read_bytes()).hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = read_corpus(args.base)
    teacher = read_corpus(args.teacher, required=False)
    merged, duplicates = merge_records(base, teacher)
    digest = write_jsonl_atomic(args.output, merged)
    print(json.dumps({
        "ok": True, "base_examples": len(base), "teacher_examples": len(teacher),
        "merged_examples": len(merged), "duplicates_skipped": duplicates,
        "output": str(Path(args.output)), "sha256": digest,
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
