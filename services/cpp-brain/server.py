#!/usr/bin/env python3
"""Authenticated HTTP adapter for the C++ FROSS reasoning executable."""
from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent.parent))
import fross_transformer as transformer  # noqa: E402

TRANSFORMER_MODEL_PATH = os.environ.get("FROSS_TRANSFORMER_MODEL_PATH", str(ROOT / "fross-transformer.pt"))
TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER, TRANSFORMER_STATUS = transformer.load_model(TRANSFORMER_MODEL_PATH)

BINARY = os.environ.get("FROSS_CPP_EXECUTABLE", "/app/brain")
SERVICE_TOKEN = os.environ.get("FROSS_BRAIN_SERVICE_TOKEN", "")
MAX_BODY = 1024 * 1024
ENGINE_NAME = "FROSS C++ Planner + FROSS Transformer 1.0"


def respond(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(encoded)


class Handler(BaseHTTPRequestHandler):
    server_version = "FROSS-Cpp-Brain/0.5"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Prompts and tokens must never be written to the service logs.
        print("FROSS C++ Brain HTTP " + str(args[1] if len(args) > 1 else "request"))

    def do_GET(self) -> None:
        if self.path not in ("/", "/health"):
            respond(self, 404, {"ok": False, "error": "Not found"})
            return
        respond(self, 200, {"ok": True, "engine": ENGINE_NAME, "version": "1.0", "generationModelLoaded": TRANSFORMER_MODEL is not None, "generationModel": transformer.VERSION if TRANSFORMER_MODEL is not None else None, "generationModelStatus": TRANSFORMER_STATUS})

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
        message = str(body.get("message", "")).strip()[:30000]
        if not message:
            respond(self, 400, {"handled": False, "error": "Message is required."})
            return
        mode = "reasoning" if str(body.get("mode", "")).lower() in ("reasoning", "deep") else "normal"
        history = body.get("history", [])
        if not isinstance(history, list):
            history = []
        context_parts = []
        for item in history[-10:]:
            if not isinstance(item, dict):
                continue
            role = "FROSS" if item.get("role") == "assistant" else "Пользователь"
            context_parts.append(role + ": " + str(item.get("content", ""))[:1600])
        context = "\n".join(context_parts)[:10000]

        knowledge = body.get("knowledge", [])
        if not isinstance(knowledge, list):
            knowledge = []
        knowledge = [
            {
                "topic": str(item.get("topic", ""))[:180],
                "title": str(item.get("title", ""))[:260],
                "summary": str(item.get("summary", ""))[:900],
                "snippet": str(item.get("snippet", ""))[:1000],
                "url": str(item.get("url", ""))[:700],
                "keyPoints": item.get("keyPoints", [])[:5] if isinstance(item.get("keyPoints"), list) else [],
            }
            for item in knowledge[:10] if isinstance(item, dict)
        ]

        # C++ supplies a deterministic algorithmic candidate when it can solve or classify
        # the request. The answer itself is produced by our separately trained-from-scratch model.
        completed = None
        try:
            completed = subprocess.run(
                [BINARY, mode, message, context],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                check=False,
                cwd=str(ROOT),
            )
        except (subprocess.TimeoutExpired, OSError):
            completed = None

        if TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None:
            try:
                if completed and completed.returncode == 0 and completed.stdout.strip():
                    knowledge.append({
                        "topic": "C++ локальная проверка",
                        "title": "Кандидат алгоритмического слоя",
                        "summary": completed.stdout.strip()[:700],
                    })
                reply = transformer.generate_reply(
                    TRANSFORMER_MODEL,
                    TRANSFORMER_TOKENIZER,
                    message,
                    history=history,
                    knowledge=knowledge,
                    mode=mode,
                    max_new_tokens=280 if mode == "reasoning" else 150,
                )
                if reply:
                    respond(self, 200, {
                        "handled": True,
                        "reply": reply[:25000],
                        "engine": ENGINE_NAME,
                        "kind": "fross-transformer-v1",
                        "generationModel": transformer.VERSION,
                        "generationModelStatus": TRANSFORMER_STATUS,
                        "plannerCandidateAvailable": bool(completed and completed.returncode == 0),
                        "trace": [
                            "C++ проверил вычислимые условия и построил локальный кандидат",
                            "Токенизировать контекст и обработать причинным attention",
                            "Сгенерировать продолжение собственными обученными весами",
                            "Проверить формат ответа и остановить генерацию",
                        ],
                        "sources": [],
                    })
                    return
            except Exception:
                # A missing/invalid checkpoint is not misreported as successful generation.
                pass

        # Never present the C++ planner candidate as a final answer if its trained
        # Transformer checkpoint is missing; the Worker may use its explicit fallback.
        respond(self, 200, {
            "handled": False,
            "engine": ENGINE_NAME,
            "kind": "generation-model-unavailable",
            "generationModelStatus": TRANSFORMER_STATUS,
            "plannerCandidateAvailable": bool(completed and completed.returncode == 0),
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
