#!/usr/bin/env python3
"""HTTP route regression tests for the FROSS Python brain service."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import tempfile
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "services" / "python-brain" / "server.py"
_spec = importlib.util.spec_from_file_location("fross_python_brain_server_routes", SERVER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load the FROSS Python brain server module.")
brain = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = brain
_spec.loader.exec_module(brain)


class PythonBrainRouteTests(unittest.TestCase):
    def test_inference_service_refuses_training_routes(self) -> None:
        token = "route-test-token"
        with patch.object(brain, "SERVICE_TOKEN", token), patch.object(brain, "TRAINING_SERVICE_MODE", False):
            request = urllib.request.Request(
                self.base_url + "/admin/training/start",
                data=b'{"steps":25000,"scheduleSteps":100000}',
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
                method="POST",
            )
            try:
                urllib.request.urlopen(request, timeout=3)
                status, body = 200, b""
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read()
            self.assertEqual(status, 503)
            self.assertIn(b"disabled on the inference service", body)

    def test_training_service_requires_persistent_volume(self) -> None:
        token = "route-test-token"
        with patch.object(brain, "SERVICE_TOKEN", token), \
             patch.object(brain, "TRAINING_SERVICE_MODE", True), \
             patch.object(brain, "TRAINING_STORAGE_CONFIRMED", False):
            request = urllib.request.Request(
                self.base_url + "/admin/training/start",
                data=b'{"steps":25000,"scheduleSteps":100000}',
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
                method="POST",
            )
            try:
                urllib.request.urlopen(request, timeout=3)
                status, body = 200, b""
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read()
            self.assertEqual(status, 503)
            self.assertIn(b"FROSS_TRAINING_STORAGE_CONFIRMED", body)

    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), brain.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=3)

    def get(self, path: str):
        request = urllib.request.Request(self.base_url + path, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def test_browser_get_on_configured_chat_url_returns_health_not_404(self) -> None:
        for path in ("/chat", "/chat?probe=1", "/chat/", "/chat/?probe=1", "/health/", "/"):
            with self.subTest(path=path):
                status, data = self.get(path)
                self.assertEqual(status, 200)
                self.assertTrue(data["ok"])
        status, data = self.get("/chat")
        self.assertIn("requestInfo", data)
        self.assertIn("POST", data["requestInfo"])

    def test_unknown_route_returns_helpful_json_without_query_string(self) -> None:
        status, data = self.get("/missing-route?private=value")
        self.assertEqual(status, 404)
        self.assertEqual(data["path"], "/missing-route")
        self.assertEqual(data["method"], "GET")
        self.assertIn("GET /health", data["routes"])
        self.assertNotIn("private=value", json.dumps(data))

    def test_query_and_trailing_slash_on_admin_post_are_normalized(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/admin/training/start/?probe=1",
            data=b'{"steps":25000,"scheduleSteps":100000}',
            headers={"Content-Type": "application/json", "Authorization": "Bearer unused"},
            method="POST",
        )
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(brain, "TRAINING_SERVICE_MODE", True), \
             patch.object(brain, "TRAINING_STORAGE_CONFIRMED", True), \
             patch.object(brain, "TRAINING_DATA_DIR", Path(folder)), \
             patch.object(brain, "TRAINING_OUTPUT_PATH", Path(folder) / "checkpoint.pt"), \
             patch.object(brain, "TRAINING_STATE_PATH", Path(folder) / "status.json"), \
             patch.object(brain, "SERVICE_TOKEN", ""):
            try:
                urllib.request.urlopen(request, timeout=3)
                status = 200
                body = b""
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read()
        self.assertEqual(status, 503)
        self.assertIn(b"Service token is not configured", body)
        self.assertNotIn(b"Not found", body)

    def test_legacy_checkpoint_does_not_deadlock_training_start_or_status(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            legacy_checkpoint = Path(folder) / "old-checkpoint.pt"
            legacy_checkpoint.write_bytes(b"legacy")
            finished = threading.Event()
            original_status = dict(brain.TRAINING_STATUS)

            def fake_train_model(**kwargs):
                finished.set()
                return {
                    "ok": True, "training_steps": 25000, "steps_this_run": 25000,
                    "train_loss": 1.0, "validation_loss": 1.1,
                    "vocab_size": 8000, "context_size": 4096, "parameters": 64819040, "cancelled": False,
                }

            try:
                with patch.object(brain, "SERVICE_TOKEN", "route-test-token"), \
                     patch.object(brain, "TRAINING_SERVICE_MODE", True), \
                     patch.object(brain, "TRAINING_STORAGE_CONFIRMED", True), \
                     patch.object(brain, "TRAINING_DATA_DIR", Path(folder)), \
                     patch.object(brain, "TRAINING_OUTPUT_PATH", legacy_checkpoint), \
                     patch.object(brain, "TRAINING_STATE_PATH", Path(folder) / "status.json"), \
                     patch("torch.load", return_value={"format_version": 1, "metrics": {}}), \
                     patch.object(brain.transformer, "train_model", side_effect=fake_train_model):
                    request = urllib.request.Request(
                        self.base_url + "/admin/training/start",
                        data=b'{"steps":25000,"scheduleSteps":100000}',
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": "Bearer route-test-token",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=3) as response:
                        self.assertEqual(response.status, 202)
                        data = json.loads(response.read().decode("utf-8"))
                    self.assertTrue(data["ok"])
                    self.assertTrue(finished.wait(3), "background training worker did not reach train_model")
                    deadline = time.monotonic() + 3
                    status_data = {}
                    while time.monotonic() < deadline:
                        request_status = urllib.request.Request(
                            self.base_url + "/admin/training/status",
                            headers={"Authorization": "Bearer route-test-token"},
                        )
                        with urllib.request.urlopen(request_status, timeout=1) as response:
                            status_data = json.loads(response.read().decode("utf-8"))
                        if status_data.get("state") == "completed":
                            break
                        time.sleep(0.02)
                    self.assertEqual(status_data.get("state"), "completed")
                    self.assertTrue(any("старая контрольная точка" in item for item in status_data.get("logs", [])))
            finally:
                with brain.TRAINING_LOCK:
                    brain.TRAINING_STATUS.clear()
                    brain.TRAINING_STATUS.update(original_status)
                    brain.TRAINING_CANCEL.clear()

    def test_sentencepiece_progress_callback_does_not_deadlock(self) -> None:
        original_status = dict(brain.TRAINING_STATUS)
        finished = threading.Event()

        def report_progress() -> None:
            brain._training_progress({
                "phase": "tokenizer_native",
                "step": 0,
                "target_steps": 25000,
                "preparation_step": 0,
                "preparation_total": 1,
                "vocab_size": 0,
                "device": "cpu",
            })
            brain._training_progress({
                "phase": "initializing_model",
                "step": 0,
                "target_steps": 25000,
                "vocab_size": 8000,
                "device": "cpu",
            })
            brain._training_progress({
                "phase": "training",
                "step": 0,
                "target_steps": 25000,
                "vocab_size": 8000,
                "device": "cpu",
            })
            finished.set()

        try:
            worker = threading.Thread(target=report_progress, daemon=True)
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(
                worker.is_alive(),
                "SentencePiece progress callback deadlocked while writing its log entry",
            )
            self.assertTrue(finished.is_set())
            with brain.TRAINING_LOCK:
                logs = list(brain.TRAINING_STATUS.get("logs", []))
                self.assertEqual(brain.TRAINING_STATUS.get("phase"), "training")
                self.assertEqual(brain.TRAINING_STATUS.get("step"), 0)
            self.assertTrue(any("SentencePiece" in line for line in logs))
            self.assertTrue(any("первый шаг" in line for line in logs))
        finally:
            with brain.TRAINING_LOCK:
                brain.TRAINING_STATUS.clear()
                brain.TRAINING_STATUS.update(original_status)

    def test_head_health_probe_is_supported(self) -> None:
        request = urllib.request.Request(self.base_url + "/health/", method="HEAD")
        with urllib.request.urlopen(request, timeout=3) as response:
            self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
