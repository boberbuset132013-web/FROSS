#!/usr/bin/env python3
"""Authenticated HTTP service for the original FROSS Transformer trained from random weights."""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
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

TRANSFORMER_MODEL_PATH = os.environ.get("FROSS_TRANSFORMER_MODEL_PATH", str(Path(tempfile.gettempdir()) / "fross-transformer.pt"))
ONNX_MODEL_PATH = os.environ.get("FROSS_ONNX_MODEL_PATH", str(Path(tempfile.gettempdir()) / "fross-transformer.onnx"))
MAX_MODEL_UPLOAD_BYTES = 90 * 1024 * 1024
MODEL_UPLOAD_REPOSITORY = os.environ.get("FROSS_MODEL_GITHUB_REPOSITORY", "boberbuset132013-web/FROSS").strip()
MODEL_UPLOAD_MIN_STEPS = max(3000, int(os.environ.get("FROSS_MODEL_MIN_STEPS", "3000")))
MODEL_UPLOAD_VOCAB_SIZE = int(os.environ.get("FROSS_MODEL_VOCAB_SIZE", "5000"))


def _bootstrap_latest_github_checkpoint(destination: str | Path) -> str | None:
    """Download the latest persisted GitHub Release asset on startup, if one exists."""
    target = Path(destination)
    if target.is_file():
        try:
            import torch
            existing = torch.load(target, map_location="cpu", weights_only=True)
            metrics = existing.get("metrics") if isinstance(existing, dict) else None
            config = existing.get("config") if isinstance(existing, dict) else None
            compatible = (
                isinstance(metrics, dict)
                and isinstance(config, dict)
                and existing.get("version") == transformer.VERSION
                and int(existing.get("format_version", 0)) in {1, 2}
                and int(metrics.get("training_steps", 0)) >= MODEL_UPLOAD_MIN_STEPS
                and int(metrics.get("vocab_size", 0)) == MODEL_UPLOAD_VOCAB_SIZE
                and metrics.get("vocab_target_reached") is True
                and int(config.get("block_size", 0)) == 320
            )
            if compatible:
                return None
        except Exception:
            pass
        # Never let a stale/incompatible baked-in checkpoint prevent recovery
        # from the latest known-good GitHub Release asset.
        target.unlink(missing_ok=True)
        Path(ONNX_MODEL_PATH).unlink(missing_ok=True)
        print("FROSS startup: existing checkpoint is incompatible; trying the latest GitHub model.", flush=True)
    repository = os.environ.get("FROSS_MODEL_GITHUB_REPOSITORY", "boberbuset132013-web/FROSS").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        return None
    partial = target.with_name(target.name + ".github-part")
    try:
        api = "https://api.github.com/repos/" + repository + "/releases/latest"
        request = urllib.request.Request(api, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "FROSS-Model-Bootstrap/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                release = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print("FROSS startup: no GitHub model release yet; waiting for upload in Settings.", flush=True)
                return None
            raise
        assets = release.get("assets", [])
        asset = next((item for item in assets if isinstance(item, dict) and item.get("name") == "fross-transformer.pt"), None)
        download_url = asset.get("browser_download_url") if isinstance(asset, dict) else None
        expected_digest = asset.get("digest") if isinstance(asset, dict) else None
        if not isinstance(download_url, str) or not download_url.startswith("https://github.com/"):
            print("FROSS startup: latest GitHub release has no model asset; waiting for upload in Settings.", flush=True)
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(urllib.request.Request(
            download_url, headers={"User-Agent": "FROSS-Model-Bootstrap/1.0", "Accept": "application/octet-stream"}
        ), timeout=90) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MODEL_UPLOAD_BYTES:
                    raise ValueError("GitHub model asset exceeds the 90-MiB limit.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total < 1_000_000:
            raise ValueError("GitHub model asset is too small to be a checkpoint.")
        actual_digest = "sha256:" + digest.hexdigest()
        if isinstance(expected_digest, str) and expected_digest and expected_digest.lower() != actual_digest.lower():
            raise ValueError("GitHub model asset SHA-256 does not match the published digest.")

        import torch
        checkpoint = torch.load(partial, map_location="cpu", weights_only=True)
        metrics = checkpoint.get("metrics") if isinstance(checkpoint, dict) else None
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if not isinstance(metrics, dict) or not isinstance(config, dict):
            raise ValueError("GitHub model checkpoint has no valid metadata.")
        if checkpoint.get("version") != transformer.VERSION or int(checkpoint.get("format_version", 0)) not in {1, 2}:
            raise ValueError("GitHub model checkpoint version or format is incompatible.")
        if int(metrics.get("training_steps", 0)) < MODEL_UPLOAD_MIN_STEPS:
            raise ValueError("GitHub model checkpoint does not meet the 3000-step minimum.")
        if int(metrics.get("vocab_size", 0)) != MODEL_UPLOAD_VOCAB_SIZE or metrics.get("vocab_target_reached") is not True:
            raise ValueError("GitHub model checkpoint does not have the required 5000-token vocabulary.")
        if int(config.get("block_size", 0)) != 320:
            raise ValueError("GitHub model checkpoint context size is not 320.")

        os.replace(partial, target)
        # A cached ONNX graph may belong to older weights. Rebuild it only after
        # the new checkpoint has passed metadata and checksum validation.
        Path(ONNX_MODEL_PATH).unlink(missing_ok=True)
        model_source = "github-release:" + str(release.get("tag_name", "latest"))
        try:
            (ROOT / "fross-model-source.txt").write_text(model_source + "\n", encoding="utf-8")
        except OSError:
            pass
        print("FROSS startup: downloaded and verified GitHub model asset; SHA-256 " + digest.hexdigest(), flush=True)
        return model_source
    except Exception as exc:
        print("FROSS startup: GitHub model bootstrap unavailable (" + type(exc).__name__ + "); upload a checkpoint from Settings.", flush=True)
        return None
    finally:
        partial.unlink(missing_ok=True)


TRANSFORMER_MODEL = None
TRANSFORMER_TOKENIZER = None
TRANSFORMER_STATUS: dict[str, Any] = {"loaded": False, "reason": "startup-pending"}
MODEL_LOAD_STATE = "pending"
MODEL_LOAD_ERROR: str | None = None
NATIVE_RUNTIME = None
CPP_RUNTIME_STATUS = "pending"

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

def _load_startup_model_background() -> None:
    """Load the persisted checkpoint after the HTTP listener is bound.

    Keeping network I/O and ONNX initialization out of module import means a
    successful container build can start its health endpoint before model assets
    are downloaded. The model remains unavailable to chat until validation passes.
    """
    global TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER, TRANSFORMER_STATUS
    global MODEL_SOURCE, MODEL_LOAD_STATE, MODEL_LOAD_ERROR
    global NATIVE_RUNTIME, CPP_RUNTIME_STATUS

    with GENERATION_LOCK:
        if TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None:
            MODEL_LOAD_STATE = "ready"
            MODEL_LOAD_ERROR = None
            return
        MODEL_LOAD_STATE = "loading"
        MODEL_LOAD_ERROR = None

    print("FROSS startup: background model loading started after HTTP server bind.", flush=True)
    candidate_native = None
    try:
        downloaded_source = None
        for attempt in range(1, 4):
            downloaded_source = _bootstrap_latest_github_checkpoint(TRANSFORMER_MODEL_PATH)
            # An existing verified checkpoint can be reused even when GitHub is
            # temporarily unavailable; otherwise retry the remote fetch briefly.
            if downloaded_source or Path(TRANSFORMER_MODEL_PATH).is_file():
                break
            if attempt < 3:
                delay = min(2 ** attempt, 8)
                print(
                    "FROSS startup: no local checkpoint yet; retrying GitHub fetch in "
                    + str(delay) + "s (" + str(attempt) + "/3).",
                    flush=True,
                )
                time.sleep(delay)
        if downloaded_source:
            MODEL_SOURCE = downloaded_source

        model, tokenizer, status = transformer.load_model(TRANSFORMER_MODEL_PATH)
        if model is None or tokenizer is None:
            reason = str(status.get("reason", "checkpoint-unavailable")) if isinstance(status, dict) else "checkpoint-unavailable"
            raise RuntimeError("No compatible checkpoint is available (" + reason + ").")

        loaded_steps = int(status.get("training_steps", 0))
        loaded_vocab = int(status.get("vocab_size", tokenizer.vocab_size))
        loaded_context = int(model.config.get("block_size", 0))
        if (
            loaded_steps < MODEL_UPLOAD_MIN_STEPS
            or loaded_vocab != MODEL_UPLOAD_VOCAB_SIZE
            or tokenizer.vocab_size != MODEL_UPLOAD_VOCAB_SIZE
            or loaded_context != 320
        ):
            raise ValueError(
                "Checkpoint requirements mismatch: training_steps="
                + str(loaded_steps) + ", vocab_size=" + str(loaded_vocab)
                + ", block_size=" + str(loaded_context) + "."
            )

        candidate_cpp_status = "disabled"
        if os.environ.get("FROSS_CPP_RUNTIME_ENABLED", "0").strip().lower() in {"1", "true", "yes"}:
            try:
                from native_runtime import NativeRuntime
                cpp_library = os.environ.get("FROSS_CPP_RUNTIME_LIBRARY", str(ROOT / "libfross_cpp.so"))
                # A newly-downloaded checkpoint invalidates any old ONNX graph.
                if not Path(ONNX_MODEL_PATH).is_file():
                    transformer.export_onnx_model(TRANSFORMER_MODEL_PATH, ONNX_MODEL_PATH)
                candidate_native = NativeRuntime(
                    cpp_library,
                    ONNX_MODEL_PATH,
                    block_size=int(model.config["block_size"]),
                    vocab_size=tokenizer.vocab_size,
                )
                candidate_cpp_status = "loaded"
            except Exception as native_error:
                # The Python/PyTorch path remains available if the optional native runtime fails.
                candidate_cpp_status = type(native_error).__name__
                if candidate_native is not None:
                    candidate_native.close()
                    candidate_native = None
                print(
                    "FROSS startup: native runtime unavailable; using Python/PyTorch ("
                    + candidate_cpp_status + ").",
                    flush=True,
                )

        with GENERATION_LOCK:
            # If a user uploaded a checkpoint while the background download was
            # running, keep that newly activated model instead of overwriting it.
            if TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None:
                if candidate_native is not None:
                    candidate_native.close()
                    candidate_native = None
                MODEL_LOAD_STATE = "ready"
                MODEL_LOAD_ERROR = None
                return

            TRANSFORMER_MODEL = model
            TRANSFORMER_TOKENIZER = tokenizer
            TRANSFORMER_STATUS = {
                **status,
                "loaded": True,
                "path": str(TRANSFORMER_MODEL_PATH),
            }
            NATIVE_RUNTIME = candidate_native
            candidate_native = None
            CPP_RUNTIME_STATUS = candidate_cpp_status
            MODEL_LOAD_STATE = "ready"
            MODEL_LOAD_ERROR = None
            try:
                (ROOT / "fross-model-source.txt").write_text(MODEL_SOURCE + "\n", encoding="utf-8")
            except OSError:
                pass

        print(
            "FROSS startup: model loaded successfully; source="
            + str(MODEL_SOURCE) + "; steps=" + str(loaded_steps)
            + "; vocab=" + str(loaded_vocab) + ".",
            flush=True,
        )
    except Exception as exc:
        if candidate_native is not None:
            candidate_native.close()
        with GENERATION_LOCK:
            # A concurrent upload may have successfully activated a model even
            # if the background fetch failed. Never replace that success with an error.
            if TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None:
                MODEL_LOAD_STATE = "ready"
                MODEL_LOAD_ERROR = None
            else:
                TRANSFORMER_MODEL = None
                TRANSFORMER_TOKENIZER = None
                TRANSFORMER_STATUS = {
                    "loaded": False,
                    "reason": "startup-load-failed",
                    "error": type(exc).__name__,
                    "path": str(TRANSFORMER_MODEL_PATH),
                }
                MODEL_LOAD_STATE = "failed"
                MODEL_LOAD_ERROR = type(exc).__name__
                if CPP_RUNTIME_STATUS == "pending":
                    CPP_RUNTIME_STATUS = "unavailable"
        print(
            "FROSS startup: model is not ready ("
            + type(exc).__name__ + "). The HTTP service remains available.",
            flush=True,
        )





class GitHubAPIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


MODEL_UPLOAD_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#101116"><title>FROSS · Загрузка модели</title>
<style>:root{color-scheme:dark;font:16px system-ui,sans-serif;background:#101116;color:#f4f5f8}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px}main{width:min(720px,100%);background:#191d25;border:1px solid #343b49;border-radius:20px;padding:26px}h1{margin:0 0 8px;font-size:25px}p{color:#bac3d3;line-height:1.5}</style></head>
<body><main><h1>FROSS · Загрузка модели</h1>
<p>Загрузка checkpoint выполняется через сайт FROSS после входа в аккаунт.</p>
<p>Открой FROSS → <b>Настройки → Загрузить модель FROSS</b>, выбери файл <b>fross-transformer.pt</b> (до 90 МБ) и дождись проверки.</p>
<p>Эта служебная страница не принимает файлы напрямую. Сервер принимает загрузку только от авторизованного Worker.</p>
</main></body></html>"""


def _send_html(handler: BaseHTTPRequestHandler, content: str) -> None:
    payload = content.encode("utf-8")
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(payload)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        return


def _github_json(method: str, url: str, token: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "FROSS-Model-Uploader/1.0",
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
            return json.loads(payload.decode("utf-8")) if payload else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read(3000).decode("utf-8", errors="replace")
        raise GitHubAPIError(exc.code, "GitHub API HTTP " + str(exc.code) + ": " + detail[:800]) from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GitHub API request failed: " + type(exc).__name__) from exc


def _upload_release_asset(upload_url: str, file_path: Path, token: str) -> dict[str, Any]:
    base_url = upload_url.split("{", 1)[0]
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError("GitHub returned an invalid release upload URL.")
    request_path = parsed.path + "?name=fross-transformer.pt"
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=900)
    size = file_path.stat().st_size
    try:
        connection.putrequest("POST", request_path)
        connection.putheader("Authorization", "Bearer " + token)
        connection.putheader("Accept", "application/vnd.github+json")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(size))
        connection.putheader("User-Agent", "FROSS-Model-Uploader/1.0")
        connection.putheader("X-GitHub-Api-Version", "2022-11-28")
        connection.endheaders()
        with file_path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        response = connection.getresponse()
        raw = response.read(5000)
        if response.status not in (200, 201):
            detail = raw.decode("utf-8", errors="replace")
            raise GitHubAPIError(response.status, "Release asset upload HTTP " + str(response.status) + ": " + detail[:800])
        return json.loads(raw.decode("utf-8"))
    finally:
        connection.close()


def _publish_checkpoint_to_github(
    file_path: Path,
    *,
    sha256: str,
    training_steps: int,
    vocab_size: int,
    github_token: str = "",
) -> tuple[dict[str, Any], str]:
    token = (github_token or os.environ.get("FROSS_GITHUB_TOKEN", "")).strip()
    repo = MODEL_UPLOAD_REPOSITORY
    if len(token) < 20:
        raise RuntimeError("FROSS_GITHUB_TOKEN is not configured in Hostless.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise RuntimeError("FROSS_MODEL_GITHUB_REPOSITORY is invalid.")

    tag = "fross-model-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha256[:8]
    release = _github_json("POST", "https://api.github.com/repos/" + repo + "/releases", token, {
        "tag_name": tag,
        "name": "FROSS Transformer 1.4 model " + tag,
        "body": "Validated FROSS Transformer 1.4 checkpoint. SHA-256: " + sha256
                + "; training steps: " + str(training_steps)
                + "; vocabulary: " + str(vocab_size) + ".",
        "draft": False,
        "prerelease": False,
        "make_latest": "true",
    })
    upload_url = str(release.get("upload_url", ""))
    if not upload_url:
        raise RuntimeError("GitHub did not return a release asset upload URL.")
    asset = _upload_release_asset(upload_url, file_path, token)
    download_url = asset.get("browser_download_url")
    if not isinstance(download_url, str) or not download_url.startswith("https://github.com/"):
        raise RuntimeError("GitHub did not publish a valid model download URL.")
    manifest = {
        "schema_version": 1,
        "model": "FROSS Transformer 1.4",
        "file": "fross-transformer.pt",
        "tag": tag,
        "release_url": release.get("html_url", ""),
        "download_url": download_url,
        "sha256": sha256,
        "size_bytes": file_path.stat().st_size,
        "training_steps": training_steps,
        "vocab_size": vocab_size,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    # Keep the repository directory small and useful: the large binary is a Release
    # Asset, while this manifest provides a stable pointer and integrity metadata.
    manifest_url = "https://api.github.com/repos/" + repo + "/contents/model-storage/current.json"
    try:
        current = _github_json("GET", manifest_url + "?ref=main", token)
        existing_sha = current.get("sha")
    except GitHubAPIError as exc:
        if exc.status != 404:
            raise
        existing_sha = None
    encoded = base64.b64encode(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii")
    update = {
        "message": "Update FROSS model storage manifest",
        "content": encoded,
        "branch": "main",
    }
    if isinstance(existing_sha, str) and existing_sha:
        update["sha"] = existing_sha
    try:
        _github_json("PUT", manifest_url, token, update)
        manifest_warning = ""
    except Exception as exc:
        # The release asset itself is already durable and becomes the latest release;
        # report manifest problems without losing a validated checkpoint upload.
        manifest_warning = type(exc).__name__ + ": " + str(exc)[:250]
    return manifest, manifest_warning


def _check_model_checkpoint(path: Path) -> tuple[dict[str, Any], Any, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Файл не читается как безопасный PyTorch checkpoint: " + type(exc).__name__) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint должен быть объектом FROSS.")
    if checkpoint.get("version") != transformer.VERSION:
        raise ValueError("Нужна совместимая версия FROSS Transformer 1.4.")
    if int(checkpoint.get("format_version", 0)) not in {1, 2}:
        raise ValueError("Неизвестный формат checkpoint.")
    metrics = checkpoint.get("metrics")
    config = checkpoint.get("config")
    if not isinstance(metrics, dict) or not isinstance(config, dict):
        raise ValueError("В checkpoint отсутствуют метрики или конфигурация.")
    steps = int(metrics.get("training_steps", 0))
    vocab_size = int(metrics.get("vocab_size", 0))
    if steps < MODEL_UPLOAD_MIN_STEPS:
        raise ValueError("Мало шагов обучения: требуется минимум " + str(MODEL_UPLOAD_MIN_STEPS) + ", получено " + str(steps) + ".")
    if vocab_size != MODEL_UPLOAD_VOCAB_SIZE or metrics.get("vocab_target_reached") is not True:
        raise ValueError("Нужен словарь из " + str(MODEL_UPLOAD_VOCAB_SIZE) + " токенов с достигнутой целью словаря.")
    if int(config.get("block_size", 0)) != 320:
        raise ValueError("Размер контекста checkpoint не соответствует FROSS Brain 1.4 (320).")
    model, tokenizer, status = transformer.load_model(path)
    if model is None or tokenizer is None:
        raise ValueError("Не удалось загрузить модель: " + json.dumps(status, ensure_ascii=False)[:500])
    if tokenizer.vocab_size != MODEL_UPLOAD_VOCAB_SIZE:
        raise ValueError("Загруженный токенизатор имеет неверный размер словаря.")
    return {"training_steps": steps, "vocab_size": vocab_size, **metrics}, model, tokenizer


def _activate_uploaded_checkpoint(
    temp_checkpoint: Path,
    temp_onnx: Path,
    new_model: Any,
    new_tokenizer: Any,
    model_status: dict[str, Any],
    new_native_runtime: Any,
    model_source: str,
) -> None:
    global TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER, TRANSFORMER_STATUS
    global NATIVE_RUNTIME, CPP_RUNTIME_STATUS, MODEL_SOURCE
    global MODEL_LOAD_STATE, MODEL_LOAD_ERROR

    checkpoint_target = Path(TRANSFORMER_MODEL_PATH)
    onnx_target = Path(ONNX_MODEL_PATH)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    onnx_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_next = checkpoint_target.with_name(checkpoint_target.name + ".next")
    onnx_next = onnx_target.with_name(onnx_target.name + ".next")
    checkpoint_backup = checkpoint_target.with_name(checkpoint_target.name + ".previous")
    onnx_backup = onnx_target.with_name(onnx_target.name + ".previous")
    try:
        shutil.copyfile(temp_checkpoint, checkpoint_next)
        shutil.copyfile(temp_onnx, onnx_next)
        if checkpoint_target.exists():
            shutil.copyfile(checkpoint_target, checkpoint_backup)
        if onnx_target.exists():
            shutil.copyfile(onnx_target, onnx_backup)
        with GENERATION_LOCK:
            try:
                os.replace(checkpoint_next, checkpoint_target)
                os.replace(onnx_next, onnx_target)
            except Exception:
                if checkpoint_backup.exists():
                    os.replace(checkpoint_backup, checkpoint_target)
                if onnx_backup.exists():
                    os.replace(onnx_backup, onnx_target)
                raise
            old_native = NATIVE_RUNTIME
            TRANSFORMER_MODEL = new_model
            TRANSFORMER_TOKENIZER = new_tokenizer
            TRANSFORMER_STATUS = {**model_status, "loaded": True, "path": str(checkpoint_target)}
            NATIVE_RUNTIME = new_native_runtime
            CPP_RUNTIME_STATUS = "loaded" if new_native_runtime is not None else "disabled"
            MODEL_SOURCE = model_source
            MODEL_LOAD_STATE = "ready"
            MODEL_LOAD_ERROR = None
            try:
                (ROOT / "fross-model-source.txt").write_text(model_source + "\n", encoding="utf-8")
            except OSError:
                pass
        if old_native is not None and old_native is not new_native_runtime:
            old_native.close()
    finally:
        checkpoint_next.unlink(missing_ok=True)
        onnx_next.unlink(missing_ok=True)
        checkpoint_backup.unlink(missing_ok=True)
        onnx_backup.unlink(missing_ok=True)


def handle_model_upload(handler: BaseHTTPRequestHandler) -> None:
    # The public browser never uploads directly to this route. The authenticated
    # Worker forwards the shared service token, providing a server-to-server gate.
    supplied_service_token = handler.headers.get("Authorization", "")
    if not SERVICE_TOKEN:
        respond(handler, 503, {"ok": False, "error": "FROSS_BRAIN_SERVICE_TOKEN не настроен в Hostless."})
        return
    if not hmac.compare_digest(supplied_service_token, "Bearer " + SERVICE_TOKEN):
        respond(handler, 401, {"ok": False, "error": "Загрузка модели разрешена только через авторизованный FROSS Worker."})
        return

    github_token = os.environ.get("FROSS_GITHUB_TOKEN", "").strip()
    forwarded_github_token = handler.headers.get("X-FROSS-GITHUB-TOKEN", "").strip()
    if forwarded_github_token and not github_token:
        github_token = forwarded_github_token
    if len(github_token) < 20:
        respond(handler, 503, {"ok": False, "error": "GitHub-токен не передан Worker и не настроен в Hostless."})
        return

    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        content_length = 0
    if content_length < 1_000_000 or content_length > MAX_MODEL_UPLOAD_BYTES:
        respond(handler, 413, {"ok": False, "error": "Размер модели должен быть от 1 МБ до 90 МБ."})
        return
    filename = urllib.parse.unquote(handler.headers.get("X-FROSS-Filename", "fross-transformer.pt"))
    filename = Path(filename.replace("\\", "/")).name
    if not filename.lower().endswith(".pt"):
        respond(handler, 400, {"ok": False, "error": "Загрузи checkpoint с расширением .pt."})
        return

    upload_dir = Path(os.environ.get("FROSS_UPLOAD_DIR", "/tmp/fross-model-upload"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_checkpoint: Path | None = None
    temp_onnx: Path | None = None
    candidate_native = None
    try:
        with tempfile.NamedTemporaryFile(prefix="fross-upload-", suffix=".pt", dir=upload_dir, delete=False) as file:
            temp_checkpoint = Path(file.name)
            remaining = content_length
            digest = hashlib.sha256()
            written = 0
            while remaining:
                chunk = handler.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Передача оборвалась до окончания файла.")
                file.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                remaining -= len(chunk)
            file.flush()
            os.fsync(file.fileno())
        if written != content_length or written < 1_000_000:
            raise ValueError("Получен неполный или слишком маленький файл.")

        metrics, new_model, new_tokenizer = _check_model_checkpoint(temp_checkpoint)
        temp_onnx = temp_checkpoint.with_suffix(".onnx")
        transformer.export_onnx_model(temp_checkpoint, temp_onnx)

        cpp_enabled = os.environ.get("FROSS_CPP_RUNTIME_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
        if cpp_enabled:
            from native_runtime import NativeRuntime
            candidate_native = NativeRuntime(
                os.environ.get("FROSS_CPP_RUNTIME_LIBRARY", str(ROOT / "libfross_cpp.so")),
                str(temp_onnx),
                block_size=int(new_model.config["block_size"]),
                vocab_size=new_tokenizer.vocab_size,
            )
            # Exercise the exact new ONNX graph before publishing it as the latest model.
            probe = new_tokenizer.encode("Проверка модели FROSS")
            logits = candidate_native.forward(probe)
            if len(logits) != new_tokenizer.vocab_size or not all(__import__("math").isfinite(value) for value in logits):
                candidate_native.close()
                candidate_native = None
                raise ValueError("Новая ONNX-модель не прошла проверку C++ runtime.")

        sha256 = digest.hexdigest()
        manifest, manifest_warning = _publish_checkpoint_to_github(
            temp_checkpoint,
            sha256=sha256,
            training_steps=int(metrics["training_steps"]),
            vocab_size=int(metrics["vocab_size"]),
            github_token=github_token,
        )
        _activate_uploaded_checkpoint(
            temp_checkpoint,
            temp_onnx,
            new_model,
            new_tokenizer,
            metrics,
            candidate_native,
            "github-release:" + str(manifest.get("tag", "latest")),
        )
        candidate_native = None
        respond(handler, 200, {
            "ok": True,
            "message": "Модель проверена, сохранена в GitHub Releases и активирована без пересборки Docker.",
            "training_steps": int(metrics["training_steps"]),
            "vocab_size": int(metrics["vocab_size"]),
            "sha256": sha256,
            "release_url": manifest.get("release_url", ""),
            "download_url": manifest.get("download_url", ""),
            "manifest_warning": manifest_warning,
        })
    except Exception as exc:
        if candidate_native is not None:
            candidate_native.close()
        status = 400 if isinstance(exc, (ValueError,)) else 502 if isinstance(exc, (GitHubAPIError, RuntimeError)) else 500
        respond(handler, status, {"ok": False, "error": str(exc)[:800], "errorType": type(exc).__name__})
    finally:
        if temp_onnx is not None:
            temp_onnx.unlink(missing_ok=True)
        if temp_checkpoint is not None:
            temp_checkpoint.unlink(missing_ok=True)


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



# Training runs independently of the browser request. Status is held in the service
# process so closing the dashboard does not cancel an active run.
TRAINING_OUTPUT_PATH = Path(os.environ.get(
    "FROSS_TRAINING_OUTPUT_PATH", str(Path(tempfile.gettempdir()) / "fross-training-output.pt")
))
TRAINING_LOCK = threading.Lock()
TRAINING_CANCEL = threading.Event()
TRAINING_STATUS: dict[str, Any] = {
    "ok": True, "state": "idle", "step": 0, "targetSteps": 600,
    "trainLoss": None, "validationLoss": None, "logs": [],
    "checkpointAvailable": TRAINING_OUTPUT_PATH.is_file(),
}
def _training_log(message: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    with TRAINING_LOCK:
        logs = TRAINING_STATUS.setdefault("logs", [])
        logs.append("[" + stamp + "] " + str(message)[:300])
        del logs[:-120]
def _training_progress(progress: dict[str, Any]) -> None:
    with TRAINING_LOCK:
        TRAINING_STATUS["step"] = int(progress.get("step", TRAINING_STATUS.get("step", 0)))
        TRAINING_STATUS["targetSteps"] = int(progress.get("target_steps", TRAINING_STATUS.get("targetSteps", 600)))
        TRAINING_STATUS["trainLoss"] = progress.get("train_loss")
        TRAINING_STATUS["validationLoss"] = progress.get("validation_loss")
        TRAINING_STATUS["bestCheckpointStep"] = int(progress.get("best_checkpoint_step", 0))
        TRAINING_STATUS["device"] = str(progress.get("device", "cpu"))
        step = int(TRAINING_STATUS["step"])
        target = max(1, int(TRAINING_STATUS["targetSteps"]))
    if step == 1 or step % 10 == 0 or step == target:
        _training_log(
            "Шаг " + str(step) + "/" + str(target)
            + " · train loss " + format(float(progress.get("train_loss", 0.0)), ".4f")
            + " · validation loss "
            + (format(float(progress["validation_loss"]), ".4f") if progress.get("validation_loss") is not None else "ещё не измерен")
        )
def _run_training(target_steps: int, schedule_steps: int) -> None:
    try:
        corpus = ROOT / "data" / "fross_corpus.jsonl"
        resume = False
        if TRAINING_OUTPUT_PATH.is_file():
            try:
                import torch
                saved = torch.load(TRAINING_OUTPUT_PATH, map_location="cpu", weights_only=True)
                resume = (
                    isinstance(saved, dict)
                    and int(saved.get("format_version", 0)) == 2
                    and isinstance(saved.get("resume_state"), dict)
                )
                if not resume:
                    _training_log("Найден старый checkpoint без состояния оптимизатора; запускаю новый этап с нуля. Старый файл будет заменён только после успешного сохранения.")
            except Exception as checkpoint_error:
                _training_log("Старый checkpoint не удалось проверить (" + type(checkpoint_error).__name__ + "); запускаю этап с нуля, не удаляя файл до сохранения.")
        _training_log("Запуск обучения Transformer 1.4; режим: " + ("продолжение" if resume else "с нуля") + ".")
        result = transformer.train_model(
            corpus_path=corpus,
            output_path=TRAINING_OUTPUT_PATH,
            steps=target_steps,
            batch_size=2,
            block_size=320,
            target_vocab_size=5000,
            seed=1337,
            resume=resume,
            schedule_steps=schedule_steps,
            progress_callback=_training_progress,
            stop_callback=TRAINING_CANCEL.is_set,
            force_cpu=True,
        )
        cancelled = bool(result.get("cancelled"))
        with TRAINING_LOCK:
            TRAINING_STATUS.update({
                "state": "cancelled" if cancelled else "completed", "step": int(result.get("training_steps", target_steps)),
                "targetSteps": target_steps, "trainLoss": result.get("train_loss"),
                "validationLoss": result.get("validation_loss"),
                "metrics": result, "checkpointAvailable": TRAINING_OUTPUT_PATH.is_file(),
                "finishedAt": datetime.now(timezone.utc).isoformat(),
            })
        _training_log("Обучение отменено; checkpoint сохранён для продолжения." if cancelled else "Обучение завершено. Контрольная точка сохранена.")
    except Exception as exc:
        with TRAINING_LOCK:
            TRAINING_STATUS.update({"state": "failed", "error": type(exc).__name__ + ": " + str(exc)[:240]})
        _training_log("Ошибка обучения: " + type(exc).__name__ + ": " + str(exc)[:180])
def _training_snapshot() -> dict[str, Any]:
    with TRAINING_LOCK:
        snapshot = dict(TRAINING_STATUS)
        snapshot["logs"] = list(TRAINING_STATUS.get("logs", []))
        snapshot["checkpointAvailable"] = TRAINING_OUTPUT_PATH.is_file()
        return snapshot

class Handler(BaseHTTPRequestHandler):
    server_version = "FROSS-Hybrid-Transformer/1.4"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Never log prompts, history, knowledge notes, credentials, or generated text.
        print("FROSS Python Transformer HTTP " + str(args[1] if len(args) > 1 else "request"))

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/admin/training/status":
            if not SERVICE_TOKEN:
                respond(self, 503, {"ok": False, "error": "Service token is not configured."})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
                respond(self, 401, {"ok": False, "error": "Unauthorized"})
                return
            respond(self, 200, _training_snapshot())
            return
        if path == "/admin/model-upload":
            _send_html(self, MODEL_UPLOAD_HTML)
            return
        if path == "/admin/training/model":
            if not SERVICE_TOKEN:
                respond(self, 503, {"ok": False, "error": "Service token is not configured."})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
                respond(self, 401, {"ok": False, "error": "Unauthorized"})
                return
            with TRAINING_LOCK:
                training_state = str(TRAINING_STATUS.get("state", "idle"))
            if training_state == "running":
                respond(self, 409, {"ok": False, "error": "Сначала дождись завершения обучения, затем скачай контрольную точку."})
                return
            if not TRAINING_OUTPUT_PATH.is_file():
                respond(self, 404, {"ok": False, "error": "Сохранённая модель не найдена. Сначала запусти и заверши обучение."})
                return
            try:
                model_file = TRAINING_OUTPUT_PATH.open("rb")
            except OSError:
                respond(self, 500, {"ok": False, "error": "Не удалось открыть файл модели."})
                return
            with model_file:
                file_size = os.fstat(model_file.fileno()).st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", 'attachment; filename="fross-transformer.pt"')
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    while True:
                        chunk = model_file.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        if path not in ("/health", "/", "/chat"):
            respond(self, 404, {
                "ok": False,
                "error": "Not found",
                "method": "GET",
                "path": path,
                "routes": ["GET /", "GET /health", "GET /chat", "GET /admin/training/status", "GET /admin/training/model"]
            })
            return
        health_data = {
            "ok": True,
            "engine": ENGINE_NAME,
            "version": "1.4",
            "modelSource": MODEL_SOURCE,
            "modelLoadState": MODEL_LOAD_STATE,
            "modelLoadError": MODEL_LOAD_ERROR,
            "generationModelLoaded": TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None,
            "generationModel": transformer.VERSION if TRANSFORMER_MODEL is not None else None,
            "generationModelStatus": TRANSFORMER_STATUS,
            "device": TRANSFORMER_STATUS.get("device", "cpu"),
            "maxGenerationTokens": GENERATION_TOKEN_LIMIT,
            "maxReplyChars": MAX_REPLY_CHARS,
            "inferenceBackend": "cpp-onnx" if NATIVE_RUNTIME is not None else "python-torch",
            "cppRuntimeLoaded": NATIVE_RUNTIME is not None,
            "cppRuntimeStatus": CPP_RUNTIME_STATUS,
        }
        if path == "/chat":
            health_data["requestInfo"] = {
                "GET": "Диагностика сервера Python brain. Для готовности модели смотри generationModelLoaded.",
                "POST": "Генерация ответа FROSS; требуется Authorization: Bearer <FROSS_BRAIN_SERVICE_TOKEN> и JSON с message.",
                "health": "/health",
                "routes": ["/chat", "/health", "/admin/training/status", "/admin/training/start", "/admin/training/model"]
            }
        respond(self, 200, health_data)

    def do_HEAD(self) -> None:
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/", "/health", "/chat"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/admin/training/cancel":
            if not SERVICE_TOKEN:
                respond(self, 503, {"ok": False, "error": "Service token is not configured."})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
                respond(self, 401, {"ok": False, "error": "Unauthorized"})
                return
            with TRAINING_LOCK:
                if TRAINING_STATUS.get("state") != "running":
                    respond(self, 409, {"ok": False, "error": "Обучение сейчас не выполняется."})
                    return
                TRAINING_CANCEL.set()
                TRAINING_STATUS["cancelRequested"] = True
            _training_log("Получен запрос на отмену; остановка после текущего шага.")
            respond(self, 202, {"ok": True, "state": "cancelling"})
            return
        if path == "/admin/training/start":
            if not SERVICE_TOKEN:
                respond(self, 503, {"ok": False, "error": "Service token is not configured."})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
                respond(self, 401, {"ok": False, "error": "Unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8192:
                    raise ValueError("Request body size is invalid.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                target_steps = int(payload.get("steps", 600))
                schedule_steps = int(payload.get("scheduleSteps", 10000))
                if target_steps not in (600, 3000, 5000, 8000, 10000):
                    raise ValueError("Choose 600, 3000, 5000, 8000 or 10000 training steps.")
                if schedule_steps != 10000 or schedule_steps < target_steps:
                    raise ValueError("Schedule must stay fixed at 10000 steps so training can resume consistently.")
            except Exception as exc:
                respond(self, 400, {"ok": False, "error": str(exc)[:240]})
                return
            with TRAINING_LOCK:
                if TRAINING_STATUS.get("state") in ("running", "cancelling"):
                    respond(self, 409, {"ok": False, "error": "Обучение уже выполняется или останавливается."})
                    return
                has_checkpoint = TRAINING_OUTPUT_PATH.is_file()
                resumable_checkpoint = False
                existing_steps = 0
                if has_checkpoint:
                    try:
                        import torch
                        existing = torch.load(TRAINING_OUTPUT_PATH, map_location="cpu", weights_only=True)
                        resumable_checkpoint = (
                            isinstance(existing, dict)
                            and int(existing.get("format_version", 0)) == 2
                            and isinstance(existing.get("resume_state"), dict)
                        )
                        if resumable_checkpoint:
                            existing_steps = int(existing.get("metrics", {}).get("training_steps", 0))
                            if target_steps <= existing_steps:
                                respond(self, 409, {"ok": False, "error": "В контрольной точке уже " + str(existing_steps) + " шагов. Выбери цель больше."})
                                return
                        else:
                            _training_log("Обнаружена несовместимая или старая контрольная точка; этот запуск начнётся с нуля.")
                    except Exception as exc:
                        respond(self, 409, {"ok": False, "error": "Не удалось проверить контрольную точку (" + type(exc).__name__ + ")."})
                        return
                TRAINING_CANCEL.clear()
                TRAINING_STATUS.update({
                    "state": "running", "step": int(existing_steps if resumable_checkpoint else 0),
                    "targetSteps": target_steps, "trainLoss": None, "validationLoss": None,
                    "error": None, "metrics": None, "startedAt": datetime.now(timezone.utc).isoformat(),
                })
            threading.Thread(
                target=_run_training, args=(target_steps, schedule_steps),
                name="fross-transformer-training", daemon=True,
            ).start()
            respond(self, 202, {"ok": True, "state": "running", "targetSteps": target_steps})
            return
        if path == "/admin/model-upload":
            handle_model_upload(self)
            return
        if path not in ("/chat", "/ready"):
            respond(self, 404, {
                "handled": False,
                "error": "Not found",
                "method": "POST",
                "path": path,
                "routes": ["POST /chat", "POST /ready", "POST /admin/training/start", "POST /admin/model-upload"]
            })
            return
        if not SERVICE_TOKEN:
            respond(self, 503, {"handled": False, "error": "Service token is not configured."})
            return
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + SERVICE_TOKEN):
            respond(self, 401, {"handled": False, "error": "Unauthorized"})
            return

        # This authenticated readiness route proves Worker/Hostless token agreement
        # and confirms model status without generating text or consuming inference time.
        if path == "/ready":
            respond(self, 200, {
                "ok": True,
                "engine": ENGINE_NAME,
                "version": "1.4",
                "modelSource": MODEL_SOURCE,
            "modelLoadState": MODEL_LOAD_STATE,
            "modelLoadError": MODEL_LOAD_ERROR,
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
                "modelLoadState": MODEL_LOAD_STATE,
                "modelLoadError": MODEL_LOAD_ERROR,
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
    http_server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("FROSS Python Brain: HTTP listener is bound; model will load in background.", flush=True)
    threading.Thread(
        target=_load_startup_model_background,
        name="fross-model-bootstrap",
        daemon=True,
    ).start()
    http_server.serve_forever()
