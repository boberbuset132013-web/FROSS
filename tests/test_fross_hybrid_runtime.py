from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import fross_transformer as brain


class HybridRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="fross-hybrid-test-")
        cls.directory = Path(cls.temp.name)
        cls.corpus = cls.directory / "corpus.jsonl"
        examples = [
            ("Привет", "Привет! Я на связи. Что хочешь обсудить?"),
            ("Что такое нейросеть?", "Нейросеть учится находить закономерности на примерах, изменяя числовые параметры."),
            ("Что такое Python?", "Python — язык программирования для скриптов, веб-сервисов и анализа данных."),
            ("Что такое JavaScript?", "JavaScript используют для интерактивных сайтов и серверных программ."),
            ("Почему модель ошибается?", "Модель может ошибаться из-за ограниченных данных, обучения и контекста."),
            ("Как учится модель?", "Модель предсказывает следующий токен, измеряет ошибку и обновляет параметры."),
            ("Что делает attention?", "Attention помогает учитывать связи между токенами в контексте."),
            ("Что такое API?", "API описывает способ взаимодействия между программами."),
            ("Что такое C++?", "C++ — компилируемый язык программирования с контролем над памятью."),
            ("Что такое трансформер?", "Трансформер обрабатывает последовательность токенов и предсказывает продолжение."),
            ("Как исправить ошибку в коде?", "Изучи сообщение об ошибке и проверь минимальный пример."),
            ("Что такое обучение?", "Обучение изменяет параметры модели по примерам и функции потерь."),
        ]
        cls.corpus.write_text(
            "\n".join(json.dumps({"user": q, "assistant": a}, ensure_ascii=False) for q, a in examples) + "\n",
            encoding="utf-8",
        )
        cls.checkpoint = cls.directory / "model.pt"
        cls.metrics = brain.train_model(
            cls.corpus, cls.checkpoint, steps=2, batch_size=1,
            block_size=32, merge_count=8, seed=7,
        )
        cls.model, cls.tokenizer, cls.status = brain.load_model(cls.checkpoint)
        if cls.model is None or cls.tokenizer is None:
            raise RuntimeError(f"Could not load temporary checkpoint: {cls.status}")
        cls.onnx_path = cls.directory / "model.onnx"
        brain.export_onnx_model(cls.checkpoint, cls.onnx_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_trained_checkpoint_is_loadable(self) -> None:
        self.assertTrue(self.checkpoint.is_file())
        self.assertTrue(self.status["loaded"])
        self.assertGreater(self.metrics["parameters"], 0)

    def test_social_greeting_is_stable(self) -> None:
        expected = "Привет! Я на связи. Что хочешь обсудить?"
        self.assertEqual(brain.generate_reply(self.model, self.tokenizer, "Привет", max_new_tokens=8), expected)
        self.assertEqual(brain.generate_reply(self.model, self.tokenizer, "Привет, Фросс!", max_new_tokens=8), expected)

    def test_latest_question_survives_long_history(self) -> None:
        captured: dict[str, list[int]] = {}
        original_forward = self.model.forward

        def recording_forward(input_ids):
            captured["input_ids"] = input_ids[0].detach().cpu().tolist()
            return original_forward(input_ids)

        self.model.forward = recording_forward
        try:
            reply = brain.generate_reply(
                self.model,
                self.tokenizer,
                "Почему нейросеть иногда отвечает бессвязно?",
                history=[
                    {"role": "user", "content": "старое сообщение " * 80},
                    {"role": "assistant", "content": "старый ответ " * 80},
                    {"role": "user", "content": "ещё контекст " * 80},
                ],
                knowledge=[{"title": "Старая заметка", "summary": "текст " * 30}],
                max_new_tokens=8,
            )
        finally:
            self.model.forward = original_forward
        decoded = self.tokenizer.decode(captured["input_ids"], stop_at_eos=False)
        self.assertIn("Почему нейросеть", decoded)
        self.assertTrue(reply.strip())

    def test_onnx_graph_executes(self) -> None:
        import onnx
        import onnxruntime as ort
        import numpy as np

        graph = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        block_size = int(self.model.config["block_size"])
        ids = [brain.BOS_ID] + self.tokenizer.encode("Пользователь: Что такое нейросеть?\nFROSS: ")
        ids = ids[-block_size:]
        padded = ids + [brain.PAD_ID] * (block_size - len(ids))
        output = session.run(["logits"], {"input_ids": np.asarray([padded], dtype=np.int64)})[0]
        self.assertEqual(output.shape, (1, block_size, self.tokenizer.vocab_size))
        self.assertTrue(torch.isfinite(torch.from_numpy(output)).all())

    def test_cpp_logits_match_python_logits_when_runtime_is_built(self) -> None:
        library = os.environ.get("FROSS_CPP_RUNTIME_LIBRARY")
        if not library:
            self.skipTest("C++ runtime was not built in this environment")
        sys.path.insert(0, str(ROOT / "services" / "python-brain"))
        from native_runtime import NativeRuntime

        block_size = int(self.model.config["block_size"])
        ids = [brain.BOS_ID] + self.tokenizer.encode("Пользователь: Что такое нейросеть?\nFROSS: ")
        ids = ids[-block_size:]
        with torch.inference_mode():
            expected = self.model(torch.tensor([ids], dtype=torch.long))[0, -1, :].cpu()
        with NativeRuntime(
            library, str(self.onnx_path), block_size=block_size,
            vocab_size=self.tokenizer.vocab_size,
        ) as runtime:
            actual = torch.tensor(runtime.forward(ids), dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
