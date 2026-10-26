#!/usr/bin/env python3
"""Generate synthetic FROSS student examples from a local Ollama teacher."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

from prepare_corpus import read_corpus, record_key, write_jsonl_atomic

SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
                "required": ["user", "assistant"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["examples"],
    "additionalProperties": False,
}
TOPICS = [
    "русский диалог, приветствия и исправление недопонимания",
    "Python: функции, списки, словари, файлы и исключения",
    "JavaScript: массивы, события, promises и обработка ошибок",
    "HTML и CSS: адаптивность, доступность и точечные правки интерфейса",
    "машинное обучение: токенизаторы, attention, loss и чекпоинты",
    "диагностика программных ошибок, тестирование, Git и зависимости",
    "письменное изучение английского или китайского языка",
    "понимание опечаток, ограничений и контекста разговора",
    "безопасная работа с переменными окружения, секретами и HTTP-ошибками",
]

def validate_example(item):
    if not isinstance(item, dict):
        return None
    user, assistant = item.get("user"), item.get("assistant")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    user = " ".join(user.split()).strip()
    assistant = assistant.strip()
    if not (8 <= len(user) <= 1200 and 35 <= len(assistant) <= 5000):
        return None
    if user.casefold() == assistant.casefold() or assistant.count(chr(96) * 3) % 2:
        return None
    return {"user": user, "assistant": assistant}

def parse_teacher_response(text):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Teacher returned invalid JSON: {exc.msg}") from exc
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("Teacher JSON must contain an examples array.")
    return [item for value in examples if (item := validate_example(value)) is not None]

def call_teacher(base_url, model, topic, count, timeout):
    prompt = (
        f"Create exactly {count} varied, accurate, compact training examples for a small Russian-language "
        f"student assistant. Topic: {topic}. Return only the JSON schema requested. Each example has a "
        "'user' prompt and an 'assistant' answer. Vary wording and intent; use concrete examples where helpful. "
        "Do not invent current facts, sources, links or personal knowledge. Do not include secrets, personal data, "
        "dangerous instructions, filler, or claims that code was executed. Keep answers useful and self-contained."
    )
    request_body = {
        "model": model,
        "system": "You are a careful teacher generating supervised text examples for a smaller student model.",
        "prompt": prompt,
        "format": SCHEMA,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.35, "num_predict": 4096},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach local Ollama at {base_url}. Start Ollama and check the model name.") from exc
    answer = payload.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an empty response.")
    return parse_teacher_response(answer)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("FROSS_TEACHER_MODEL", "qwen2.5:3b"))
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base-corpus", default="data/fross_corpus.jsonl")
    parser.add_argument("--output", default="data/fross_teacher_corpus.jsonl")
    args = parser.parse_args()
    if not 1 <= args.count <= 1000 or not 1 <= args.batch_size <= 12 or args.timeout < 10:
        parser.error("Use count 1-1000, batch-size 1-12, and timeout >= 10.")

    root = Path(__file__).resolve().parents[2]
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    base, output = resolve(args.base_corpus), resolve(args.output)
    if not base.is_file():
        print(f"Base corpus not found: {base}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)

    output_rows = read_corpus(output, required=False)
    seen = set()
    for path in (base,):
        for row in read_corpus(path):
            seen.add(record_key(row))
    for row in output_rows:
        seen.add(record_key(row))

    created = 0
    attempts = 0
    limit = max(3, ((args.count + args.batch_size - 1) // args.batch_size) * 4)
    print(f"Local teacher model: {args.model}")
    print(f"Local API: {args.base_url}")
    print("Teacher weights are only used to generate examples; only FROSS is trained.")
    while created < args.count and attempts < limit:
        topic = TOPICS[attempts % len(TOPICS)]
        amount = min(args.batch_size, args.count - created)
        attempts += 1
        print(f"Batch {attempts}/{limit}: {topic}", flush=True)
        try:
            batch = call_teacher(args.base_url, args.model, topic, amount, args.timeout)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Teacher failed after creating {created} examples: {exc}", file=sys.stderr)
            print("Already saved examples were preserved. Re-run the command to continue.", file=sys.stderr)
            return 1
        fresh = []
        for example in batch[:amount]:
            key = record_key(example)
            if key in seen:
                continue
            seen.add(key)
            fresh.append({**example, "source": "local_teacher_distillation", "teacher_model": args.model})
        if fresh:
            # Replace atomically after every successful batch. An interrupted write
            # cannot leave a half JSONL record that prevents the next run from resuming.
            output_rows.extend(fresh)
            write_jsonl_atomic(output, output_rows)
            created += len(fresh)
            print(f"Accepted {len(fresh)}; this run: {created}/{args.count}", flush=True)
        else:
            print("No unique valid examples in this batch; trying another topic.", flush=True)

    complete = created >= args.count
    print(json.dumps({"ok": complete, "requested": args.count, "created": created,
                      "complete": complete, "teacher_model": args.model, "output": str(output)},
                     ensure_ascii=False, indent=2))
    if not complete:
        print("The requested count was not reached. Existing examples were saved; re-run to add more.", file=sys.stderr)
    return 0 if complete else 1

if __name__ == "__main__":
    raise SystemExit(main())
