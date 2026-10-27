#!/usr/bin/env python3
"""Reliably download a FROSS checkpoint from a direct URL or public Yandex Disk link."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "FROSS-Checkpoint-Downloader/1.2"
MAX_ATTEMPTS = 5
MIN_CHECKPOINT_BYTES = 1_000_000


def latest_github_checkpoint_url(repository: str) -> tuple[str, str] | None:
    """Find the newest published Release Asset, if one exists, without hard-coding a model URL."""
    repository = str(repository or "").strip()
    if not repository or "/" not in repository or " " in repository:
        return None
    api_url = "https://api.github.com/repos/" + repository + "/releases/latest"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("FROSS_GITHUB_TOKEN", "").strip()
    if len(token) >= 20:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            metadata = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise
    assets = metadata.get("assets", [])
    for asset in assets if isinstance(assets, list) else []:
        if not isinstance(asset, dict) or asset.get("name") != "fross-transformer.pt":
            continue
        url = asset.get("browser_download_url")
        if isinstance(url, str) and url.startswith("https://github.com/"):
            return url, str(metadata.get("tag_name", "latest"))
    return None


def is_yandex_public_link(url: str) -> bool:
    return url.startswith((
        "https://disk.yandex.ru/d/",
        "https://yadi.sk/d/",
        "https://disk.yandex.com/d/",
    ))


def resolve_download_url(source_url: str) -> str:
    """Resolve a fresh Yandex temporary download URL for every attempt."""
    if not is_yandex_public_link(source_url):
        return source_url

    api_url = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
        + urllib.parse.urlencode({"public_key": source_url})
    )
    request = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        metadata = json.load(response)
    download_url = metadata.get("href", "")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError("Yandex Disk did not return a valid HTTPS download URL.")
    return download_url


def download_once(source_url: str, partial_path: Path) -> int:
    # The Yandex href is short-lived, so resolve it again on every retry.
    download_url = resolve_download_url(source_url)
    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
    )
    total = 0
    with urllib.request.urlopen(request, timeout=90) as response:
        status = getattr(response, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"Download returned HTTP {status}.")
        with partial_path.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                total += len(chunk)
            file.flush()
            os.fsync(file.fileno())

    if total < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"Downloaded file is only {total} bytes; expected a checkpoint, not an HTML/error page."
        )
    return total


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download_model.py URL OUTPUT_PATH", file=sys.stderr)
        return 2

    source_url, output_name = sys.argv[1], sys.argv[2]
    repository = os.environ.get("FROSS_MODEL_GITHUB_REPOSITORY", "").strip()
    if repository:
        try:
            latest = latest_github_checkpoint_url(repository)
            if latest is not None:
                source_url, release_tag = latest
                print(f"Using published GitHub model release {release_tag} from {repository}.", flush=True)
            else:
                print(f"No published FROSS checkpoint release found in {repository}; using configured fallback URL.", flush=True)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as error:
            print(
                f"Could not query GitHub model releases ({type(error).__name__}); using configured fallback URL.",
                file=sys.stderr,
                flush=True,
            )
    output = Path(output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_name(output.name + ".part")
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        partial_path.unlink(missing_ok=True)
        try:
            total = download_once(source_url, partial_path)
            # Publish the file atomically only after a complete download.
            os.replace(partial_path, output)
            print(f"Downloaded checkpoint: {total} bytes (attempt {attempt}/{MAX_ATTEMPTS}).")
            return 0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, RuntimeError) as error:
            last_error = error
            partial_path.unlink(missing_ok=True)
            if attempt >= MAX_ATTEMPTS:
                break
            delay = min(2 ** attempt, 10)
            print(
                f"Checkpoint download attempt {attempt}/{MAX_ATTEMPTS} failed: "
                f"{type(error).__name__}: {error}. Retrying in {delay}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    partial_path.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    print(
        f"Checkpoint download failed after {MAX_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
