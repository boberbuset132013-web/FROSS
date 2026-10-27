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

TRANSFORMER_MODEL_PATH = os.environ.get("FROSS_TRANSFORMER_MODEL_PATH", str(ROOT / "fross-transformer.pt"))
ONNX_MODEL_PATH = os.environ.get("FROSS_ONNX_MODEL_PATH", str(Path(TRANSFORMER_MODEL_PATH).with_suffix(".onnx")))


def _bootstrap_latest_github_checkpoint(destination: str | Path) -> str | None:
    """Download the latest persisted GitHub Release asset on startup, if one exists."""
    target = Path(destination)
    if target.is_file():
        return None
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
                    raise ValueError("GitHub model asset exceeds the 120-MiB limit.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total < 1_000_000:
            raise ValueError("GitHub model asset is too small to be a checkpoint.")

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


_startup_model_source = _bootstrap_latest_github_checkpoint(TRANSFORMER_MODEL_PATH)
if _startup_model_source:
    MODEL_SOURCE = _startup_model_source
TRANSFORMER_MODEL, TRANSFORMER_TOKENIZER, TRANSFORMER_STATUS = transformer.load_model(TRANSFORMER_MODEL_PATH)
if TRANSFORMER_MODEL is not None and TRANSFORMER_TOKENIZER is not None:
    _loaded_steps = int(TRANSFORMER_STATUS.get("training_steps", 0))
    _loaded_vocab = int(TRANSFORMER_STATUS.get("vocab_size", TRANSFORMER_TOKENIZER.vocab_size))
    _loaded_context = int(TRANSFORMER_MODEL.config.get("block_size", 0))
    if (
        _loaded_steps < MODEL_UPLOAD_MIN_STEPS
        or _loaded_vocab != MODEL_UPLOAD_VOCAB_SIZE
        or TRANSFORMER_TOKENIZER.vocab_size != MODEL_UPLOAD_VOCAB_SIZE
        or _loaded_context != 320
    ):
        TRANSFORMER_MODEL = None
        TRANSFORMER_TOKENIZER = None
        TRANSFORMER_STATUS = {
            "loaded": False,
            "reason": "checkpoint-requirements-mismatch",
            "training_steps": _loaded_steps,
            "vocab_size": _loaded_vocab,
            "block_size": _loaded_context,
        }

NATIVE_RUNTIME = None
CPP_RUNTIME_STATUS = "disabled"
if os.environ.get("FROSS_CPP_RUNTIME_ENABLED", "0").strip().lower() in {"1", "true", "yes"}:
    try:
        from native_runtime import NativeRuntime
        CPP_RUNTIME_LIBRARY = os.environ.get("FROSS_CPP_RUNTIME_LIBRARY", str(ROOT / "libfross_cpp.so"))
        if TRANSFORMER_TOKENIZER is None or TRANSFORMER_MODEL is None:
            raise RuntimeError("trained checkpoint is not loaded")
        if not Path(ONNX_MODEL_PATH).is_file():
            transformer.export_onnx_model(TRANSFORMER_MODEL_PATH, ONNX_MODEL_PATH)
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
MAX_MODEL_UPLOAD_BYTES = 120 * 1024 * 1024
MODEL_UPLOAD_REPOSITORY = os.environ.get("FROSS_MODEL_GITHUB_REPOSITORY", "boberbuset132013-web/FROSS").strip()
MODEL_UPLOAD_MIN_STEPS = max(3000, int(os.environ.get("FROSS_MODEL_MIN_STEPS", "3000")))
MODEL_UPLOAD_VOCAB_SIZE = int(os.environ.get("FROSS_MODEL_VOCAB_SIZE", "5000"))
_UPLOAD_ATTEMPTS: dict[str, tuple[int, float]] = {}
_UPLOAD_ATTEMPTS_LOCK = threading.Lock()
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




class GitHubAPIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


MODEL_UPLOAD_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#101116"><title>FROSS · Загрузка модели</title>
<style>
:root{color-scheme:dark;font:16px system-ui,sans-serif;background:#101116;color:#f4f5f8}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px}
main{width:min(720px,100%);background:#191d25;border:1px solid #343b49;border-radius:20px;padding:26px;box-shadow:0 20px 70px #0005}
h1{margin:0 0 8px;font-size:25px}p{color:#bac3d3;line-height:1.5}
label{display:block;margin:18px 0 7px;color:#d6ddeb;font-weight:600}
input[type=password],input[type=file]{display:block;width:100%;padding:12px;border:1px solid #475265;border-radius:10px;background:#10141b;color:#fff}
button{margin-top:18px;border:0;border-radius:10px;padding:12px 16px;background:#087e4c;color:white;font-weight:700;cursor:pointer;width:100%}
button:disabled{opacity:.55;cursor:wait}.bar{height:8px;background:#303846;border-radius:9px;overflow:hidden;margin-top:16px}.bar span{display:block;width:0;height:100%;background:#20c77a;transition:width .1s}
#status{white-space:pre-wrap;overflow-wrap:anywhere;background:#11151c;border:1px solid #303846;padding:12px;border-radius:10px;margin-top:14px;min-height:44px}
small{color:#98a5b8}.warn{color:#ffd181}
</style>
</head>
<body><main>
<h1>FROSS · Загрузка модели</h1>
<p>Загрузи совместимый файл <b>fross-transformer.pt</b>. Сервер проверит Brain 1.4, словарь 5000 токенов и минимум 3000 шагов, сохранит резервную копию в GitHub и переключит модель без пересборки Docker.</p>
<p class="warn">Операция может занять до 30 минут. Не закрывай вкладку до результата. Не загружай сюда GitHub-токен — только пароль загрузки.</p>
<form id="uploadForm">
<label for="password">Пароль загрузки</label>
<input id="password" type="password" autocomplete="current-password" required>
<label for="modelFile">Файл модели (.pt, максимум 120 МБ)</label>
<input id="modelFile" type="file" accept=".pt,application/octet-stream" required>
<button id="submit" type="submit">Проверить и загрузить модель</button>
</form>
<div class="bar"><span id="bar"></span></div>
<div id="status" role="status" aria-live="polite">Ожидаю файл модели.</div>
<small>Пароль отправляется только заголовком HTTPS-запроса. Страница не сохраняет его в браузере.</small>
</main>
<script>
(function(){
const form=document.getElementById("uploadForm"),pass=document.getElementById("password"),file=document.getElementById("modelFile");
const submit=document.getElementById("submit"),status=document.getElementById("status"),bar=document.getElementById("bar");
form.addEventListener("submit",function(event){
event.preventDefault();
const picked=file.files&&file.files[0];
if(!picked){status.textContent="Сначала выбери .pt файл.";return;}
if(!picked.name.toLowerCase().endsWith(".pt")){status.textContent="Нужен файл с расширением .pt.";return;}
if(picked.size<1000000||picked.size>120*1024*1024){status.textContent="Размер файла должен быть от 1 МБ до 120 МБ.";return;}
submit.disabled=true;bar.style.width="0%";status.textContent="Передаю файл на сервер…";
const xhr=new XMLHttpRequest();xhr.open("POST",location.pathname);xhr.timeout=30*60*1000;
xhr.setRequestHeader("Content-Type","application/octet-stream");
xhr.setRequestHeader("X-FROSS-Upload-Password",pass.value);
xhr.setRequestHeader("X-FROSS-Filename",encodeURIComponent(picked.name));
xhr.upload.onprogress=function(e){if(e.lengthComputable){const pct=Math.round(e.loaded/e.total*100);bar.style.width=pct+"%";status.textContent=pct>=100?"Файл отправлен. Проверяю контрольную сумму, модель и сохраняю копию в GitHub…":"Передача файла: "+pct+"%";}};
xhr.onload=function(){let data={};try{data=JSON.parse(xhr.responseText||"{}");}catch(_e){}
if(xhr.status>=200&&xhr.status<300&&data.ok){bar.style.width="100%";status.textContent="ГОТОВО\n"+(data.message||"Модель активирована.")+"\nШаги: "+data.training_steps+"\nСловарь: "+data.vocab_size+"\nSHA-256: "+data.sha256+"\nGitHub: "+(data.release_url||"сохранено");if(data.manifest_warning)status.textContent+="\nПредупреждение манифеста: "+data.manifest_warning;}
else{status.textContent="Ошибка HTTP "+xhr.status+": "+(data.error||xhr.responseText||"Не удалось загрузить модель.");}
submit.disabled=false;};
xhr.onerror=function(){status.textContent="Сетевая ошибка. Проверь доступность сервиса; не отправляй файл повторно, пока не проверишь /health.";submit.disabled=false;};
xhr.ontimeout=function(){status.textContent="Запрос превысил 30 минут. Проверь /health и GitHub Releases, прежде чем повторять загрузку.";submit.disabled=false;};
xhr.send(picked);
});
})();
</script></body></html>"""


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
    expected_password = os.environ.get("FROSS_MODEL_UPLOAD_PASSWORD", "")
    github_token = os.environ.get("FROSS_GITHUB_TOKEN", "").strip()
    forwarded_github_token = handler.headers.get("X-FROSS-GITHUB-TOKEN", "").strip()
    # When GitHub credentials are forwarded from the Cloudflare Worker, require
    # the existing brain service token so arbitrary callers cannot inject their token.
    if forwarded_github_token and not github_token:
        supplied_service_token = handler.headers.get("Authorization", "")
        if not SERVICE_TOKEN or not hmac.compare_digest(
            supplied_service_token, "Bearer " + SERVICE_TOKEN
        ):
            respond(handler, 401, {"ok": False, "error": "Upload proxy authentication failed."})
            return
        github_token = forwarded_github_token
    if not expected_password:
        respond(handler, 503, {"ok": False, "error": "Секрет FROSS_MODEL_UPLOAD_PASSWORD не настроен в Hostless."})
        return
    if len(github_token) < 20:
        respond(handler, 503, {"ok": False, "error": "GitHub-токен не передан Worker и не настроен в Hostless."})
        return

    peer = str(handler.client_address[0]) if handler.client_address else "unknown"
    now = time.time()
    with _UPLOAD_ATTEMPTS_LOCK:
        attempts, last_attempt = _UPLOAD_ATTEMPTS.get(peer, (0, 0.0))
        if now - last_attempt > 900:
            attempts = 0
        if attempts >= 5:
            respond(handler, 429, {"ok": False, "error": "Слишком много неверных паролей. Подожди 15 минут и повтори."})
            return
        supplied = handler.headers.get("X-FROSS-Upload-Password", "")
        if not hmac.compare_digest(supplied, expected_password):
            _UPLOAD_ATTEMPTS[peer] = (attempts + 1, now)
            respond(handler, 401, {"ok": False, "error": "Неверный пароль загрузки."})
            return
        _UPLOAD_ATTEMPTS.pop(peer, None)

    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        content_length = 0
    if content_length < 1_000_000 or content_length > MAX_MODEL_UPLOAD_BYTES:
        respond(handler, 413, {"ok": False, "error": "Размер модели должен быть от 1 МБ до 120 МБ."})
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


class Handler(BaseHTTPRequestHandler):
    server_version = "FROSS-Hybrid-Transformer/1.4"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Never log prompts, history, knowledge notes, credentials, or generated text.
        print("FROSS Python Transformer HTTP " + str(args[1] if len(args) > 1 else "request"))

    def do_GET(self) -> None:
        if self.path == "/admin/model-upload":
            _send_html(self, MODEL_UPLOAD_HTML)
            return
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
        if self.path == "/admin/model-upload":
            handle_model_upload(self)
            return
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
