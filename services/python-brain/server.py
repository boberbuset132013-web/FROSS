#!/usr/bin/env python3
"""Authenticated HTTP service for the original FROSS Transformer trained from random weights."""
from __future__ import annotations

import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if not (ROOT / "fross_transformer.py").exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    MODEL_SOURCE = (ROOT / "fross-model-source.txt").read_text(encoding="utf-8").strip()[:100]
except OSError:
    MODEL_SOURCE = "unknown"

import fross_transformer as transformer  # noqa: E402
# Runtime containers may expose many host CPUs despite a small CPU quota.
# Cap PyTorch's thread pools before the first model load or inference.
transformer.configure_runtime_threads()

TRANSFORMER_MODEL_PATH = os.environ.get("FROSS_TRANSFORMER_MODEL_PATH", str(ROOT / "fross-transformer.pt"))
TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER, TRANSFORMER_STATUS = transformer.load_model(TRANSFORMER_MODEL_PATH)

NATIVE_RUNTIME = None
CPP_RUNTIME_STATUS = "disabled"
if os.environ.get("FROSS_CPP_RUNTIME_ENABLED", "0").strip().lower() in {"1", "true", "yes"}:
    try:
        from native_runtime import NativeRuntime
        ONNX_MODEL_PATH = os.environ.get(
            "FROSS_ONNX_MODEL_PATH",
            str(Path(TRANSFORMER_MODEL_PATH).with_suffix(".onnx")),
        )
        CPP_RUNTIME_LIBRARY = os.environ.get("FROSS_CPP_RUNTIME_LIBRARY", str(ROOT / "libfross_cpp.so"))
        if TRANSFORMER_TOKENIZER is None or TRANSFORMER_MODEL is None:
            raise RuntimeError("trained checkpoint is not loaded")
        NATIVE_RUNTIME = NativeRuntime(
            CPP_RUNTIME_LIBRARY,
            ONNX_MODEL_PATH,
            block_size=int(TRANSFORMER_MODEL.config["block_size"]),
            vocab_size=TRANSFORMER_TOKENIZER.vocab_size,
        )
        CPP_RUNTIME_STATUS = "loaded"
    except Exception as exc:
        # Keep the service available on the known-good Python path if the native
        # runtime cannot be loaded; the health endpoint makes this visible.
        CPP_RUNTIME_STATUS = type(exc).__name__

MAX_BODY = 1024 * 1024
SERVICE_TOKEN = os.environ.get("FROSS_BRAIN_SERVICE_TOKEN", "")
ENGINE_NAME = "FROSS Hybrid Transformer Brain 1.4"
MAX_REPLY_CHARS = 50000
try:
    GENERATION_TOKEN_LIMIT = max(
        1,
        min(transformer.MAX_GENERATION_TOKENS, int(os.environ.get("FROSS_MAX_GENERATION_TOKENS", "5000"))),
    )
except (TypeError, ValueError):
    GENERATION_TOKEN_LIMIT = transformer.MAX_GENERATION_TOKENS
# Serialize inference so concurrent visitors do not multiply activation memory on a small server.
GENERATION_LOCK = threading.Lock()


def respond(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    """Send JSON while treating a client disconnect as a normal closed request."""
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        # A Worker, proxy, or browser may time out/disconnect while inference is
        # running. The response can no longer be delivered; don't print a traceback
        # or let this routine create noisy server errors.
        return


class Handler(BaseHTTPRequestHandler):
    server_version = "FROSS-Hybrid-Transformer/1.4"

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
            "version": "1.4",
            "modelSource": MODEL_SOURCE,
            "generationModelLoaded": TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None,
            "generationModel": transformer.VERSION if TRANSFORMER_MODEL is not None else None,
            "generationModelStatus": TRANSFORMER_STATUS,
            "device": TRANSFORMER_STATUS.get("device", "cpu"),
            "maxGenerationTokens": GENERATION_TOKEN_LIMIT,
            "maxReplyChars": MAX_REPLY_CHARS,
            "inferenceBackend": "cpp-onnx" if NATIVE_RUNTIME is not None else "python-torch",
            "cppRuntimeLoaded": NATIVE_RUNTIME is not None,
            "cppRuntimeStatus": CPP_RUNTIME_STATUS,
        })

    def do_POST(self) -> None:
        if self.path not in ("/chat", "/ready"):
            respond(self, 404, {"handled": False, "error": "Not found"})
            return
        if not SERVICE_TOKEN:
            respond(self, 503, {"handled": False, "error": "Service token is not configured."})
            return
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
            respond(self, 401, {"handled": False, "error": "Unauthorized"})
            return

        # This authenticated readiness route proves Worker/Hostless token agreement
        # and confirms model status without generating text or consuming inference time.
        if self.path == "/ready":
            respond(self, 200, {
                "ok": True,
                "engine": ENGINE_NAME,
                "version": "1.4",
                "modelSource": MODEL_SOURCE,
                "authenticated": True,
                "generationModelLoaded": TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None,
                "generationModel": transformer.VERSION if TRANSFORMER_MODEL is not None else None,
                "generationModelStatus": TRANSFORMER_STATUS,
                "inferenceBackend": "cpp-onnx" if NATIVE_RUNTIME is not None else "python-torch",
                "cppRuntimeLoaded": NATIVE_RUNTIME is not None,
                "cppRuntimeStatus": CPP_RUNTIME_STATUS,
            })
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
        # Do not let a low-CPU host spend minutes generating one HTTP response.
        # The Worker sends an explicit budget; direct callers get a safe mode-based default.
        default_request_tokens = 512 if mode == "reasoning" else 256
        try:
            requested_tokens = int(body.get("max_new_tokens", default_request_tokens))
        except (TypeError, ValueError):
            requested_tokens = default_request_tokens
        request_token_limit = max(1, min(requested_tokens, GENERATION_TOKEN_LIMIT))

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
                "generationModel": transformer.VERSION,
                "generationModelStatus": TRANSFORMER_STATUS,
            })
            return

        try:
            with GENERATION_LOCK:
                reply = transformer.generate_reply(
                    TRANSFORMER_MODEL,
                    TRANSFORMER_TOKENIZER,
                    message,
                    history=history,
                    knowledge=knowledge,
                    mode=mode,
                    max_new_tokens=request_token_limit,
                    native_runtime=NATIVE_RUNTIME,
                ).strip()
        except Exception as exc:
            # Return only the exception class, never the prompt or internal traceback.
            respond(self, 200, {
                "handled": False,
                "engine": ENGINE_NAME,
                "kind": "generation-error",
                "diagnostic": type(exc).__name__,
                "generationModel": transformer.VERSION,
                "generationModelStatus": TRANSFORMER_STATUS,
            })
            return

        if not reply:
            respond(self, 200, {
                "handled": False,
                "engine": ENGINE_NAME,
                "kind": "empty-generation",
                "generationModel": transformer.VERSION,
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
            "trace": (
                [
                    "1/14 — определить цель и ожидаемый результат",
                    "2/14 — выделить условия и запреты",
                    "3/14 — учесть историю диалога",
                    "4/14 — разделить задачу на подзадачи",
                    "5/14 — отметить известные факты",
                    "6/14 — определить пробелы в знаниях",
                    "7/14 — использовать поиск при необходимости",
                    "8/14 — извлечь факты из источников",
                    "9/14 — сравнить надёжность источников",
                    "10/14 — выбрать способ решения",
                    "11/14 — сформулировать ответ своими словами",
                    "12/14 — проверить логику и крайние случаи",
                    "13/14 — проверить ограничения и формат",
                    "14/14 — выдать вывод и обозначить неопределённость",
                ]
                if mode == "reasoning" else
                [
                    "1/7 — разобрать точную формулировку",
                    "2/7 — определить намерение пользователя",
                    "3/7 — учесть историю и ограничения",
                    "4/7 — найти нужные факты и пробелы",
                    "5/7 — проверить термины и источники при необходимости",
                    "6/7 — сформировать прямой ответ своими словами",
                    "7/7 — проверить релевантность и формат",
                ]
            ),
            "sources": [],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
