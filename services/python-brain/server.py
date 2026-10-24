#!/usr/bin/env python3
"""Authenticated HTTP service for the original FROSS Transformer trained from random weights."""
from __future__ import annotations

import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if not (ROOT / "fross_transformer.py").exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import fross_transformer as transformer  # noqa: E402

TRANSFORMER_MODEL_PATH = os.environ.get("FROSS_TRANSFORMER_MODEL_PATH", str(ROOT / "fross-transformer.pt"))
TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER, TRANSFORMER_STATUS = transformer.load_model(TRANSFORMER_MODEL_PATH)

MAX_BODY = 1024 * 1024
SERVICE_TOKEN = os.environ.get("FROSS_BRAIN_SERVICE_TOKEN", "")
ENGINE_NAME = "FROSS Python Transformer Brain 1.0"
MAX_REPLY_CHARS = 25000


def respond(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    server_version = "FROSS-Python-Transformer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Never log prompts, history, knowledge notes, credentials, or generated text.
        print("FROSS Python Transformer HTTP " + str(args[1] if len(args) > 1 else "request"))

    def do_GET(self) -> None:
        if self.path not in ("/health", "/"):
            respond(self, 404, {"ok": False, "error": "Not found"})
            return
        respond(self, 200, {
            "ok": True,
            "engine": ENGINE_NAME,
            "version": "1.0",
            "generationModelLoaded": TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None,
            "generationModel": transformer.VERSION if TRANSFORMER_MODEL is not None else None,
            "generationModelStatus": TRANSFORMER_STATUS,
        })

    def do_POST(self) -> None:
        if self.path != "/chat":
            respond(self, 404, {"handled": False, "error": "Not found"})
            return
        if not SERVICE_TOKEN:
            respond(self, 503, {"handled": False, "error": "Service token is not configured."})
            return
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
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
        message = raw_message.strip()
        mode = "reasoning" if str(body.get("mode", "normal")).lower() in ("reasoning", "deep") else "normal"

        raw_history = body.get("history", [])
        if not isinstance(raw_history, list):
            raw_history = []
        history = [
            {
                "role": "assistant" if item.get("role") == "assistant" else "user",
                "content": str(item.get("content", ""))[:1600],
            }
            for item in raw_history[-10:] if isinstance(item, dict)
        ]

        raw_knowledge = body.get("knowledge", [])
        if not isinstance(raw_knowledge, list):
            raw_knowledge = []
        knowledge = []
        for item in raw_knowledge[:24]:
            if not isinstance(item, dict):
                continue
            points = item.get("keyPoints", [])
            knowledge.append({
                "topic": str(item.get("topic", ""))[:180],
                "title": str(item.get("title", ""))[:260],
                "url": str(item.get("url", ""))[:700],
                "summary": str(item.get("summary", ""))[:900],
                "snippet": str(item.get("snippet", ""))[:1800],
                "keyPoints": [str(point)[:300] for point in points[:5]] if isinstance(points, list) else [],
                "source": str(item.get("source", ""))[:80],
            })

        if TRANSFORMER_MODEL is None or TRANSFORMER_TOKENIZER is None:
            respond(self, 200, {
                "handled": False,
                "engine": ENGINE_NAME,
                "kind": "generation-model-unavailable",
                "generationModelStatus": TRANSFORMER_STATUS,
            })
            return

        try:
            reply = transformer.generate_reply(
                TRANSFORMER_MODEL,
                TRANSFORMER_TOKENIZER,
                message,
                history=history,
                knowledge=knowledge,
                mode=mode,
                max_new_tokens=360 if mode == "reasoning" else 180,
            ).strip()
        except Exception as exc:
            # Return only the exception class, never the prompt or internal traceback.
            respond(self, 200, {
                "handled": False,
                "engine": ENGINE_NAME,
                "kind": "generation-error",
                "diagnostic": type(exc).__name__,
                "generationModelStatus": TRANSFORMER_STATUS,
            })
            return

        if not reply:
            respond(self, 200, {
                "handled": False,
                "engine": ENGINE_NAME,
                "kind": "empty-generation",
                "generationModelStatus": TRANSFORMER_STATUS,
            })
            return

        respond(self, 200, {
            "handled": True,
            "reply": reply[:MAX_REPLY_CHARS],
            "engine": ENGINE_NAME,
            "kind": "fross-transformer-v1",
            "generationModel": transformer.VERSION,
            "generationModelStatus": TRANSFORMER_STATUS,
            "trace": [
                "Разбить запрос и контекст на собственные BPE-токены",
                "Обработать последовательность блоками причинного self-attention",
                "Предсказать следующий токен обученными весами и повторить цикл",
                "Сохранить абзацы и форматирование сгенерированного текста",
            ],
            "sources": [],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
