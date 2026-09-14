#!/usr/bin/env python3
"""Fetch the public, pinned model and verify every byte before installation.

    python fetch_doclaynet_model.py --output-root runtime/doclaynet

The output contains model/ and model-manifest.json. Existing files are reused
only when their size and SHA-256 match the bundled lock. Requests inherits proxy
settings, including HTTPS_PROXY; redirected or signed URLs are never logged.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import requests


CHUNK_SIZE = 1024 * 1024
MODEL_ID = "Aryn/deformable-detr-DocLayNet"
REVISION = "d5503a90ae08dd43565de6984a5dd7924cad2400"
REQUIRED_FILES = {"README.md", "config.json", "preprocessor_config.json", "model.safetensors"}


class DownloadError(Exception):
    """An error whose message contains no remote URL or exception detail."""

    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def load_lock(path):
    raw = path.read_bytes()
    try:
        lock = json.loads(raw)
    except (ValueError, UnicodeError):
        raise DownloadError("Cannot parse the bundled model lock.") from None
    if not isinstance(lock, dict):
        raise DownloadError("Invalid bundled model lock.")
    if (lock.get("schema_version") != "doclaynet-model-lock-v1"
            or lock.get("model_id") != MODEL_ID
            or lock.get("revision") != REVISION
            or lock.get("license") != "Apache-2.0"
            or lock.get("training_data") != "DocLayNet; exact training split list is not published"
            or lock.get("training_overlap_status") != "unknown"):
        raise DownloadError("The bundled model lock has an unexpected identity or schema.")
    files = lock.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise DownloadError("The bundled model lock has an unexpected file set.")
    for spec in files.values():
        if (not isinstance(spec, dict)
                or type(spec.get("size")) is not int or spec["size"] <= 0
                or not isinstance(spec.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
            raise DownloadError("The bundled model lock has invalid file checksums or sizes.")
    return lock, hashlib.sha256(raw).hexdigest()


def matches(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == spec["sha256"]


def download_once(session, url, destination, spec):
    """Leave an existing destination intact until a replacement is verified."""
    temporary = None
    try:
        with session.get(url, stream=True, timeout=(20, 120),
                         headers={"Accept-Encoding": "identity"}) as response:
            if response.status_code != 200:
                code = int(response.status_code)
                raise DownloadError(
                    f"HTTP {code}", retryable=code in (408, 429) or 500 <= code <= 599
                )
            digest = hashlib.sha256()
            size = 0
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".partial",
                dir=destination.parent, delete=False
            ) as target:
                temporary = Path(target.name)
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > spec["size"]:
                        raise DownloadError("download exceeds the locked size", retryable=True)
                    digest.update(chunk)
                    target.write(chunk)
                if size != spec["size"] or digest.hexdigest() != spec["sha256"]:
                    raise DownloadError("download size or SHA-256 mismatch", retryable=True)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
            temporary = None
    except requests.exceptions.Timeout:
        raise DownloadError("network timeout", retryable=True) from None
    except requests.exceptions.RequestException:
        # Requests exceptions may contain signed redirect URLs or proxy secrets.
        raise DownloadError("network or transport failure", retryable=True) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_file(session, url, destination, spec, attempts):
    if matches(destination, spec):
        print(f"VERIFIED {destination.name} (reused)", flush=True)
        return
    for attempt in range(1, attempts + 1):
        try:
            download_once(session, url, destination, spec)
            print(f"VERIFIED {destination.name} (downloaded)", flush=True)
            return
        except DownloadError as error:
            if not error.retryable or attempt == attempts:
                raise DownloadError(
                    f"{destination.name}: {error}; failed after {attempt} attempt(s)."
                ) from None
            print(f"RETRY {destination.name} ({attempt}/{attempts}): {error}",
                  file=sys.stderr, flush=True)
            time.sleep(min(2 ** (attempt - 1), 16))


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".partial",
            dir=path.parent, delete=False
        ) as target:
            temporary = Path(target.name)
            json.dump(value, target, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_model(output_root, attempts=4):
    lock, _ = load_lock(Path(__file__).with_name("doclaynet-model.lock.json"))
    model_dir = output_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}"
    with requests.Session() as session:
        # Keep Requests' trust_env default so HTTPS_PROXY and CA settings work.
        for name, spec in lock["files"].items():
            fetch_file(session, f"{base_url}/{name}", model_dir / name, spec, attempts)
    # Omit run-specific timestamps/paths so a verified rerun is byte-identical.
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "license": lock["license"],
        "files": {
            name: {"sha256": lock["files"][name]["sha256"], "size": lock["files"][name]["size"]}
            for name in ("README.md", "config.json", "preprocessor_config.json", "model.safetensors")
        },
        "training_data": lock["training_data"],
        "training_overlap_status": lock["training_overlap_status"],
    }
    atomic_json(output_root / "model-manifest.json", manifest)
    print("MODEL READY: model/ and model-manifest.json are verified.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runtime/doclaynet"))
    parser.add_argument("--attempts", type=int, default=4,
                        help="maximum attempts per file (default: 4)")
    args = parser.parse_args()
    if not 1 <= args.attempts <= 10:
        parser.error("--attempts must be between 1 and 10")
    try:
        fetch_model(args.output_root, args.attempts)
    except DownloadError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"ERROR: local filesystem failure (errno={error.errno}).", file=sys.stderr)
        return 1
    except Exception:
        # A third-party exception must not expose redirect URLs via a traceback.
        print("ERROR: model preparation failed unexpectedly; no files were accepted without verification.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
