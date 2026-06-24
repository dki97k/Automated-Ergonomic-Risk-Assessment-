#!/usr/bin/env python3
"""Download a gated Hugging Face file with robust byte-range retries."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
from huggingface_hub import get_token


def download(repo: str, filename: str, destination: Path) -> None:
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token found. Run `hf auth login` first.")

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    while True:
        offset = destination.stat().st_size if destination.exists() else 0
        headers = {"Authorization": f"Bearer {token}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=120.0),
            ) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416:
                        print(f"Download complete: {destination} ({offset} bytes)")
                        return
                    response.raise_for_status()

                    if offset and response.status_code != 206:
                        raise RuntimeError(
                            "Server did not honor the byte-range request; "
                            "refusing to overwrite the partial file."
                        )

                    expected_total = None
                    content_range = response.headers.get("content-range")
                    if content_range and "/" in content_range:
                        expected_total = int(content_range.rsplit("/", 1)[1])
                    elif not offset and response.headers.get("content-length"):
                        expected_total = int(response.headers["content-length"])

                    mode = "ab" if offset else "wb"
                    with destination.open(mode) as output:
                        for chunk in response.iter_bytes(8 * 1024 * 1024):
                            output.write(chunk)

            current_size = destination.stat().st_size
            if expected_total is None or current_size == expected_total:
                print(f"Download complete: {destination} ({current_size} bytes)")
                return
            print(f"Partial download: {current_size}/{expected_total} bytes")
        except (httpx.HTTPError, OSError) as error:
            current_size = destination.stat().st_size if destination.exists() else 0
            print(f"Download interrupted at {current_size} bytes: {error}")

        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("filename")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    download(args.repo, args.filename, args.destination)


if __name__ == "__main__":
    main()

