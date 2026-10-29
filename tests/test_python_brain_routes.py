#!/usr/bin/env python3
"""HTTP route regression tests for the FROSS Python brain service."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
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
            data=b'{"steps":600,"scheduleSteps":10000}',
            headers={"Content-Type": "application/json", "Authorization": "Bearer unused"},
            method="POST",
        )
        with patch.object(brain, "SERVICE_TOKEN", ""):
            try:
                urllib.request.urlopen(request, timeout=3)
                status = 200
                body = b""
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read()
        self.assertEqual(status, 503)
        self.assertIn(b"Service token is not configured", body)
        self.assertNotIn(b"Not found", body)

    def test_head_health_probe_is_supported(self) -> None:
        request = urllib.request.Request(self.base_url + "/health/", method="HEAD")
        with urllib.request.urlopen(request, timeout=3) as response:
            self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
