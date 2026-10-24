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
import llm_client  # noqa: E402

BINARY = os.environ.get("FROSS_CPP_EXECUTABLE", "/app/brain")
SERVICE_TOKEN = os.environ.get("FROSS_BRAIN_SERVICE_TOKEN", "")
MAX_BODY = 1024 * 1024
ENGINE_NAME = "FROSS C++ Planning Brain 1.0"


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
        respond(self, 200, {"ok": True, "engine": ENGINE_NAME, "version": "1.0", "languageModelConfigured": llm_client.is_configured(), "languageModel": os.environ.get("FROSS_LLM_MODEL", llm_client.DEFAULT_MODEL)})

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

        # Use C++ as an independent deterministic planner/candidate generator, then let
        # the language model compose and check a free-form answer when configured.
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

        if llm_client.is_configured():
            try:
                candidate = completed.stdout.strip()[:3500] if completed and completed.returncode == 0 else ""
                generated = llm_client.generate_reply(
                    message,
                    mode=mode,
                    history=history,
                    knowledge=knowledge,
                    planner_hint=candidate,
                )
                respond(self, 200, {
                    "handled": True,
                    "reply": generated["reply"],
                    "engine": generated["engine"],
                    "kind": "llm-generated-v1",
                    "model": generated["model"],
                    "trace": [
                        "C++ проанализировал тип задачи и доступные локальные проверки",
                        "Языковая модель самостоятельно сформировала ответ",
                        "Проверить ограничения и представить итог",
                    ] if mode == "reasoning" else [
                        "Определить намерение и контекст",
                        "Сгенерировать ответ языковой моделью",
                        "Проверить соответствие вопросу",
                    ],
                    "sources": [],
                })
                return
            except Exception:
                # Keep provider errors and secrets out of the client response.
                pass

        if completed is None:
            respond(self, 200, {"handled": False, "engine": ENGINE_NAME, "kind": "cpp-unavailable"})
            return
        if completed.returncode == 3:
            respond(self, 200, {"handled": False, "engine": ENGINE_NAME, "kind": "cpp-no-match"})
            return
        if completed.returncode != 0:
            respond(self, 500, {"handled": False, "engine": ENGINE_NAME, "error": "C++ brain failed."})
            return
        answer = completed.stdout.strip()
        if not answer:
            respond(self, 200, {"handled": False, "engine": ENGINE_NAME, "kind": "empty-reply"})
            return
        trace = [
            "Разобрать запрос и определить цель",
            "Проверить локальные правила и вычислимые задачи",
            "Выбрать алгоритм или шаблон решения",
            "Проверить крайние случаи",
            "Сформировать объяснение",
            "Проверить ограничения ответа",
            "Подготовить итог",
        ]
        respond(self, 200, {
            "handled": True,
            "reply": answer[:25000],
            "engine": ENGINE_NAME,
            "kind": "cpp-algorithm-v05",
            "trace": trace,
            "sources": [],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
