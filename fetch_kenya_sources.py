#!/usr/bin/env python3
"""Download the frozen Kenya sources; resume files and verify their exact bytes."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch_one(item: dict, root: Path, log_root: Path, proxy: str | None) -> dict:
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe source path: {relative}')
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = item.get('sha256')
    expected_bytes = item.get('expected_bytes', item.get('size_bytes'))
    if target.exists() and expected_hash and digest(target) == expected_hash:
        print(json.dumps({'event': 'verified_existing', 'path': str(relative)}), flush=True)
        return dict(item, sha256=expected_hash, size_bytes=target.stat().st_size)
    if target.exists() and expected_hash:
        raise RuntimeError(f'Existing final source hash mismatch: {target}; preserve and investigate it')
    part = target.with_name(target.name + '.part')
    if target.exists():
        if part.exists():
            raise RuntimeError(f'Both unverified final file and partial file exist: {target}')
        target.rename(part)
    log = log_root / (relative.as_posix().replace('/', '__') + '.log')
    command = ['curl', '-4', '-L', '--fail', '--retry', '8', '--retry-delay', '5',
               '--retry-all-errors', '--connect-timeout', '20', '--speed-time', '90',
               '--speed-limit', '1024', '--continue-at', '-', '--output', str(part)]
    if proxy:
        command += ['--proxy', proxy]
    if item.get('etag'):
        command += ['--header', 'If-Match: ' + item['etag']]
    command.append(item['url'])
    # A completed partial file may only need its verification/atomic rename.
    if not (part.exists() and expected_bytes is not None and part.stat().st_size == expected_bytes):
        print(json.dumps({'event': 'download_started', 'path': str(relative)}), flush=True)
        with log.open('ab') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    if expected_bytes is not None and part.stat().st_size != expected_bytes:
        raise RuntimeError(f'Source size mismatch: {relative}: {part.stat().st_size} != {expected_bytes}')
    actual_hash = digest(part)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f'Source SHA256 mismatch: {relative}')
    if item.get('md5_hex'):
        # GCS exposes this independently of our newly calculated SHA256.
        provider_digest = hashlib.md5(usedforsecurity=False)
        with part.open('rb') as source:
            for block in iter(lambda: source.read(8 << 20), b''):
                provider_digest.update(block)
        if provider_digest.hexdigest() != item['md5_hex']:
            raise RuntimeError(f'Provider MD5 mismatch: {relative}')
    os.replace(part, target)
    result = dict(item, sha256=actual_hash, size_bytes=target.stat().st_size)
    print(json.dumps({'event': 'download_verified', 'path': str(relative),
                      'bytes': target.stat().st_size, 'sha256': actual_hash}), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-lock', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--log-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--proxy', default=None, help='Optional network proxy, never saved in the lock')
    p.add_argument('--freeze-lock', type=Path,
                   help='Initial acquisition only: write observed SHA256 values to this new lock')
    args = p.parse_args()
    lock = json.loads(args.source_lock.read_text())
    items = [lock['boundary'], *lock['sources']]
    paths = [item['path'] for item in items]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source paths in lock')
    if not args.freeze_lock and any(not item.get('sha256') for item in items):
        raise ValueError('Reproduction requires SHA256 for every source; use --freeze-lock for initial acquisition')
    if not 1 <= args.workers <= 32:
        raise ValueError('workers must be between 1 and 32')
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, item, args.raw_root, args.log_root, args.proxy): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result['path']] = result
            receipt = args.log_root / 'download-receipt.json'
            tmp = receipt.with_suffix('.tmp')
            tmp.write_text(json.dumps({'completed': results}, indent=2) + '\n')
            os.replace(tmp, receipt)
    if args.freeze_lock:
        frozen = dict(lock)
        frozen['boundary'] = results[lock['boundary']['path']]
        frozen['sources'] = [results[item['path']] for item in lock['sources']]
        frozen['acquired_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        args.freeze_lock.parent.mkdir(parents=True, exist_ok=True)
        if args.freeze_lock.exists():
            raise FileExistsError(f'Will not overwrite an existing frozen lock: {args.freeze_lock}')
        args.freeze_lock.write_text(json.dumps(frozen, indent=2) + '\n')
    print(json.dumps({'event': 'all_sources_verified', 'source_count': len(items)}), flush=True)


if __name__ == '__main__':
    main()
