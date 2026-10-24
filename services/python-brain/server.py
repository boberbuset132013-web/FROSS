#!/usr/bin/env python3
"""FROSS Python Brain 0.5 HTTP adapter. Standard library only."""
from __future__ import annotations

import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if not (ROOT / "fross_brain_v05.py").exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import fross_brain_v05 as brain  # noqa: E402

MAX_BODY = 1024 * 1024
SERVICE_TOKEN = os.environ.get("FROSS_BRAIN_SERVICE_TOKEN", "")
ENGINE_NAME = "FROSS Python Brain 0.5"


def respond(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    server_version = "FROSS-Python-Brain/0.5"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Never log the prompt, chat history, auth header, or saved notes.
        print("FROSS Python Brain request: " + str(args[1] if len(args) > 1 else "request"))

    def do_GET(self) -> None:
        if self.path not in ("/health", "/"):
            respond(self, 404, {"ok": False, "error": "Not found"})
            return
        respond(self, 200, {"ok": True, "engine": ENGINE_NAME, "version": "0.5"})

    def do_POST(self) -> None:
        if self.path != "/chat":
            respond(self, 404, {"handled": False, "error": "Not found"})
            return
        if not SERVICE_TOKEN:
            respond(self, 503, {"handled": False, "error": "Service token is not configured."})
            return
        expected = "Bearer " + SERVICE_TOKEN
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, expected):
            respond(self, 401, {"handled": False, "error": "Unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            respond(self, 413, {"handled": False, "error": "Request body is too large or empty."})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            respond(self, 400, {"handled": False, "error": "Expected a valid JSON body."})
            return
        if not isinstance(body, dict):
            respond(self, 400, {"handled": False, "error": "Expected a JSON object."})
            return

        raw_message = str(body.get("message", ""))[:30000].strip()
        if not raw_message:
            respond(self, 400, {"handled": False, "error": "Message is required."})
            return
        message = brain.correct_text(raw_message)
        mode = "reasoning" if str(body.get("mode", "normal")).lower() in ("reasoning", "deep") else "normal"
        history = body.get("history", [])
        if not isinstance(history, list):
            history = []
        history = [
            {"role": "assistant" if item.get("role") == "assistant" else "user",
             "content": str(item.get("content", ""))[:1600]}
            for item in history[-10:] if isinstance(item, dict)
        ]
        knowledge = body.get("knowledge", [])
        if not isinstance(knowledge, list):
            knowledge = []
        knowledge = [
            {
                "topic": str(item.get("topic", ""))[:180],
                "title": str(item.get("title", ""))[:260],
                "url": str(item.get("url", ""))[:700],
                "snippet": str(item.get("snippet", ""))[:3200],
                "source": str(item.get("source", ""))[:80],
                "score": max(0.0, min(1.0, float(item.get("score", 0) or 0))),
            }
            for item in knowledge[:24] if isinstance(item, dict)
        ]

        try:
            reply = brain.quick_reply(message, is_creator=False, mode=mode)
            kind = "python-rule-v05"
            trace = [
                "Нормализовать запрос и исправить известные опечатки",
                "Проверить локальные правила и безопасный вычислитель",
                "Сформировать ответ или пример кода",
                "Проверить формат и ограничения",
            ]
            if not reply:
                result = brain.think(
                    message, history=history, results=[], mode=mode,
                    is_creator=False, knowledge=knowledge
                )
                kind = str(result.get("kind", "python-no-match"))
                if kind == "no-match-v05":
                    respond(self, 200, {
                        "handled": False, "engine": ENGINE_NAME, "kind": kind,
                        "trace": trace,
                    })
                    return
                reply = str(result.get("reply", "")).strip()
                trace = list(result.get("trace", trace))[:4]
            if not reply:
                respond(self, 200, {"handled": False, "engine": ENGINE_NAME, "kind": "empty-reply"})
                return
            respond(self, 200, {
                "handled": True,
                "reply": reply[:25000],
                "engine": ENGINE_NAME,
                "kind": kind,
                "trace": trace,
                "sources": [],
            })
        except Exception:
            # Do not send stack traces, environment variables, or secrets to the client.
            respond(self, 500, {"handled": False, "engine": ENGINE_NAME, "error": "Python brain failed."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
