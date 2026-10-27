#!/usr/bin/env python3
"""Download a FROSS checkpoint from a direct URL or a public Yandex Disk link."""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download_model.py URL OUTPUT_PATH", file=sys.stderr)
        return 2
    source_url, output_name = sys.argv[1], sys.argv[2]
    download_url = source_url

    if source_url.startswith(("https://disk.yandex.ru/d/", "https://yadi.sk/d/", "https://disk.yandex.com/d/")):
        api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?" + urllib.parse.urlencode({"public_key": source_url})
        request = urllib.request.Request(api_url, headers={"User-Agent": "FROSS-Checkpoint-Downloader/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.load(response)
        download_url = metadata.get("href", "")
        if not download_url:
            raise RuntimeError("Yandex Disk did not return a download URL.")

    output = Path(output_name)
    request = urllib.request.Request(download_url, headers={"User-Agent": "FROSS-Checkpoint-Downloader/1.0"})
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as file:
        status = getattr(response, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"Download returned HTTP {status}.")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            total += len(chunk)
    if total < 1_000_000:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is only {total} bytes; expected a model checkpoint, not an HTML/error page.")
    print(f"Downloaded checkpoint candidate: {total} bytes.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Checkpoint download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
