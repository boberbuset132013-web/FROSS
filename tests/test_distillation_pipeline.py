from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DISTILLATION = ROOT / "training" / "distillation"
sys.path.insert(0, str(DISTILLATION))

from prepare_corpus import merge_records, read_corpus, write_jsonl_atomic
from teacher_distill import parse_teacher_response


class DistillationPipelineTests(unittest.TestCase):
    def test_teacher_examples_are_deduplicated_without_dropping_context(self) -> None:
        base = [
            {"user": "Привет! ", "assistant": "Привет! Что будем делать?"},
            {"user": "Почему?", "assistant": "Уточню причину на примере.", "history": [
                {"role": "user", "content": "Объясни цикл."},
                {"role": "assistant", "content": "Цикл повторяет действие."},
            ]},
        ]
        teacher = [
            {"user": " привет! ", "assistant": "Другой ответ не нужен."},
            {"user": "Почему?", "assistant": "Потому что это другой контекст.", "history": [
                {"role": "user", "content": "Объясни функцию."},
                {"role": "assistant", "content": "Функция выполняет задачу."},
            ]},
        ]
        merged, duplicates = merge_records(base, teacher)
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["assistant"], "Привет! Что будем делать?")

    def test_committed_teacher_corpus_merges_with_original_corpus_cleanly(self) -> None:
        base = read_corpus(ROOT / "data" / "fross_corpus.jsonl")
        teacher = read_corpus(ROOT / "data" / "fross_teacher_corpus.jsonl")
        merged, duplicates = merge_records(base, teacher)
        self.assertEqual(len(base), 328)
        self.assertGreaterEqual(len(teacher), 400)
        self.assertEqual(duplicates, 0)
        self.assertEqual(len(merged), len(base) + len(teacher))

    def test_committed_teacher_examples_match_the_reported_errors(self) -> None:
        teacher = read_corpus(ROOT / "data" / "fross_teacher_corpus.jsonl")

        python_type_errors = [
            row for row in teacher
            if "TypeError: unsupported operand type(s) for +: 'int' and 'str'" in row["user"]
        ]
        self.assertEqual(len(python_type_errors), 3)
        for row in python_type_errors:
            answer = row["assistant"].lower()
            self.assertIn("typeerror", answer)
            self.assertIn("int(y)", answer)
            self.assertNotIn("nullpointerexception", answer)
            self.assertNotIn("javascript", answer)

        array_example = next(
            row for row in teacher
            if row["user"].startswith("Как обработать ошибку при работе с несуществующим элементом массива?")
        )
        self.assertIn("не выбрасывает исключение", array_example["assistant"].lower())
        self.assertIn("undefined", array_example["assistant"].lower())
        self.assertNotIn("выведет: ошибка: undefined", array_example["assistant"].lower())

        sql_example = next(
            row for row in teacher
            if row["user"].startswith("В PostgreSQL получаю ошибку")
        )
        self.assertIn("не обязано вызывать синтаксическую ошибку", sql_example["assistant"].lower())
        self.assertIn("неявный псевдоним", sql_example["assistant"].lower())

    def test_multiline_python_examples_use_copyable_fenced_blocks(self) -> None:
        teacher = read_corpus(ROOT / "data" / "fross_teacher_corpus.jsonl")
        expected_prompts = (
            "Я написал с ошибками. Попробуй понять смысл: обясни что такое цикл в питон",
            "Как поймать ошибку в Python и не уронить программу?",
            "Как прочитать JSON из файла в Python?",
            "Как проверить, существует ли файл в Python?",
        )
        for prompt in expected_prompts:
            with self.subTest(prompt=prompt):
                row = next(item for item in teacher if item["user"] == prompt)
                self.assertIn("```python\n", row["assistant"])
                self.assertIn("\n```", row["assistant"])

    def test_read_corpus_reports_bad_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.jsonl"
            path.write_text('{"user":"one","assistant":"two"}\n{broken\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"broken.jsonl:2"):
                read_corpus(path)

    def test_merge_output_is_utf8_jsonl_and_atomic_writer_returns_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "nested" / "merged.jsonl"
            rows = [{"user": "Привет", "assistant": "Привет! Чем могу помочь?"}]
            digest = write_jsonl_atomic(output, rows)
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), rows[0])

    def test_teacher_json_parser_keeps_only_valid_examples(self) -> None:
        payload = {
            "examples": [
                {"user": "Объясни, зачем нужна функция.", "assistant": "Функция объединяет действия под одним именем и может принимать параметры."},
                {"user": "Коротко", "assistant": "Да"},
                {"user": "Что-то", "assistant": " "},
                {"user": 42, "assistant": "Ответ, который не должен пройти проверку длины."},
            ]
        }
        rows = parse_teacher_response(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["user"], "Объясни, зачем нужна функция.")

    def test_teacher_json_parser_rejects_malformed_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            parse_teacher_response("{not-json")

    def test_teacher_examples_cannot_contain_unclosed_code_fence(self) -> None:
        payload = {"examples": [{
            "user": "Как выглядит пример кода?",
            "assistant": "Вот пример:\n" + chr(96) * 3 + "python\nprint('hi')",
        }]}
        self.assertEqual(parse_teacher_response(json.dumps(payload)), [])


if __name__ == "__main__":
    unittest.main()
