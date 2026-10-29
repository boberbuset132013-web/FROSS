#!/usr/bin/env python3
"""Focused tests for the FROSS Transformer implemented and trained from scratch."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

# Running this file directly sets sys.path[0] to tests/, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fross_transformer as transformer


class TransformerTests(unittest.TestCase):
    def test_default_architecture_and_generation_budget(self) -> None:
        self.assertEqual(transformer.DEFAULT_CONFIG["n_layer"], 6)
        self.assertEqual(transformer.DEFAULT_CONFIG["n_head"], 6)
        self.assertEqual(transformer.DEFAULT_CONFIG["n_embd"], 192)
        self.assertEqual(transformer.DEFAULT_CONFIG["block_size"], 320)
        self.assertEqual(transformer.VERSION, "FROSS Transformer 1.4")
        self.assertEqual(transformer.MAX_GENERATION_TOKENS, 5000)

    def test_bpe_tokenizer_round_trip(self) -> None:
        samples = [
            "Привет, FROSS!",
            "Attention использует Query, Key и Value.",
            "Скобки (2 + 3) работают.",
            "Первая строка\nВторая строка.",
        ]
        tokenizer = transformer.BPETokenizer.train(samples * 3, merge_count=32, min_frequency=2)
        for text in samples:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_brain_14_can_reach_target_vocabulary_without_merging_across_words(self) -> None:
        import itertools
        import string

        # Enough distinct word forms to test an exact target without a large training run.
        words = [f"слово{a}{b}word" for a, b in itertools.product(string.ascii_lowercase, repeat=2)]
        words.extend(["你好，世界", "谢谢你", "常见"])
        tokenizer = transformer.BPETokenizer.train(words, target_vocab_size=256)
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertTrue(tokenizer.pretokenized)
        for piece in tokenizer.pieces[4:]:
            self.assertFalse(any(ch.isspace() for ch in piece) and any(ch.isalpha() for ch in piece))
        self.assertIn("你", tokenizer.char_to_id)
        self.assertIn("好", tokenizer.char_to_id)
        sample = "Привет, FROSS!\nСлово и word42. 你好，世界"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_generation_prefix_matches_training_format(self) -> None:
        record = {"user": "Привет, FROSS.", "assistant": "Привет!"}
        formatted = transformer._format_record(record)
        prefix = transformer._format_generation_prefix(record["user"])
        self.assertEqual(transformer.RESPONSE_PREFIX, "FROSS: ")
        self.assertTrue(formatted.startswith(prefix))
        self.assertEqual(formatted[len(prefix):], "Привет!\n")

    def test_response_strategy_routes_common_user_intents(self) -> None:
        self.assertIn("короткая социальная реплика", transformer._response_strategy("Привет!"))
        self.assertIn("перевод", transformer._response_strategy("Переведи это на русский."))
        self.assertIn("ограниченная правка", transformer._response_strategy("Не меняй ничего кроме цвета."))
        self.assertIn("продолжение или исправление", transformer._response_strategy("Почему?"))
        self.assertIn("готовый результат или код", transformer._response_strategy("Напиши код на JavaScript."))

    def test_training_format_includes_dialogue_history(self) -> None:
        record = {
            "history": [
                {"role": "user", "content": "Сделай кнопку синей."},
                {"role": "assistant", "content": "Изменю фон кнопки на синий."},
            ],
            "user": "Нет, только при наведении.",
            "assistant": "Добавь синий цвет в :hover, а обычное состояние оставь прежним.",
        }
        formatted = transformer._format_record(record)
        self.assertIn("Пользователь: Сделай кнопку синей.", formatted)
        self.assertIn("FROSS: Изменю фон кнопки на синий.", formatted)
        self.assertTrue(formatted.endswith("FROSS: Добавь синий цвет в :hover, а обычное состояние оставь прежним.\n"))

    def test_corpus_loader_preserves_dialogue_history(self) -> None:
        import json
        import tempfile

        history = [{"role": "user", "content": "Оставь HTML без изменений."}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.jsonl"
            rows = [
                {"history": history, "user": "Исправь только CSS.", "assistant": "Изменю только CSS."},
                *({"user": f"вопрос {i}", "assistant": f"ответ {i}"} for i in range(7)),
            ]
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            loaded = transformer._load_corpus(path)
        self.assertEqual(loaded[0]["history"][0]["content"], "Оставь HTML без изменений.")
        self.assertEqual(loaded[0]["history"][0]["role"], "user")
        self.assertEqual(loaded[1]["history"], [])

    def test_causal_attention_cannot_see_future_tokens(self) -> None:
        torch.manual_seed(7)
        model = transformer.FrossTransformer(
            vocab_size=48,
            config={
                "block_size": 16,
                "n_embd": 16,
                "n_head": 4,
                "n_layer": 2,
                "dropout": 0.0,
            },
        )
        model.eval()
        first = torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.long)
        changed_future = torch.tensor([[8, 9, 10, 11, 12, 25]], dtype=torch.long)
        with torch.inference_mode():
            logits_a = model(first)
            logits_b = model(changed_future)
        self.assertTrue(torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6, rtol=1e-6))

    def test_model_uses_randomly_initialized_parameters(self) -> None:
        torch.manual_seed(19)
        model = transformer.FrossTransformer(
            vocab_size=32,
            config={
                "block_size": 32,
                "n_embd": 16,
                "n_head": 4,
                "n_layer": 1,
                "dropout": 0.0,
            },
        )
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 1000)
        self.assertTrue(torch.isfinite(model.token_embedding.weight).all())

    def test_clean_generated_text_preserves_code_and_paragraphs(self) -> None:
        text = "Первая строка.\n\n```python\ndef main():\n    return 1\n```\n\n\nИтог."
        cleaned = transformer._clean_generated_text(text)
        self.assertIn("\n\n```python", cleaned)
        self.assertIn("\n    return 1", cleaned)
        self.assertNotIn("\n\n\n", cleaned)


    def test_training_resumes_from_saved_optimizer_and_rng_state(self) -> None:
        import json
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus_path = root / "corpus.jsonl"
            checkpoint_path = root / "checkpoint.pt"
            records = [
                {
                    "user": f"вопрос {i}: объясни Python функции и массивы",
                    "assistant": f"ответ {i}: функция выполняет задачу и возвращает результат",
                }
                for i in range(40)
            ]
            corpus_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
                encoding="utf-8",
            )
            tiny_config = {
                "block_size": 32,
                "n_embd": 16,
                "n_head": 4,
                "n_layer": 1,
                "dropout": 0.0,
            }
            with patch.dict(transformer.DEFAULT_CONFIG, tiny_config, clear=True):
                first = transformer.train_model(
                    corpus_path,
                    checkpoint_path,
                    steps=1,
                    batch_size=1,
                    block_size=32,
                    target_vocab_size=256,
                    seed=91,
                    schedule_steps=2,
                )
                self.assertEqual(first["training_steps"], 1)
                self.assertEqual(first["steps_this_run"], 1)

                continued = transformer.train_model(
                    corpus_path,
                    checkpoint_path,
                    steps=2,
                    batch_size=1,
                    block_size=32,
                    target_vocab_size=256,
                    seed=91,
                    resume=True,
                    schedule_steps=3,
                )
                self.assertEqual(continued["training_steps"], 2)
                self.assertEqual(continued["steps_this_run"], 1)
                self.assertEqual(continued["schedule_steps"], 3)

                model, tokenizer, status = transformer.load_model(checkpoint_path)
                self.assertIsNotNone(model)
                self.assertIsNotNone(tokenizer)
                self.assertTrue(status["loaded"])


if __name__ == "__main__":
    unittest.main()
