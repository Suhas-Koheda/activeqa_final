"""Parallel-range downloader for large HF files on a throttled CDN.

Splits the file into N chunks downloaded concurrently over separate HTTP
connections (each re-resolving the redirect), retries each chunk, and finally
stitches parts into the target file. Resumable: completed chunks are cached on
disk and verified by size.

Usage:
    python scripts/parallel_hf_download.py \
        --parts 16 --url https://huggingface.co/OWNER/REPO/resolve/main/FILE \
        --out models/x.bin
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

TEMP = ".ranges"


def head_size(url: str) -> int:
    out = subprocess.run(
        ["curl", "-sIL", url],
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("x-linked-size:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"no x-linked-size header for {url}")


def fetch_chunk(url: str, start: int, end: int, path: Path, retries: int = 30) -> None:
    """Download one byte range into its part file (skip if already complete)."""
    expected = end - start + 1
    for attempt in range(retries):
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            return
        if have > expected or (attempt > 0 and have == 0):
            path.unlink(missing_ok=True)
            have = 0
        if have == 0:
            header = ["-r", f"{start}-{end}"]
        else:
            header = ["-r", f"{start + have}-{end}"]
        result = None
        mode = "wb" if have == 0 else "ab"
        range_header = [f"{start}-{end}"] if have == 0 else [f"{start + have}-{end}"]
        with open(path, mode) as stream:
            result = subprocess.run(
                ["curl", "-sL", "--fail", "-r", *range_header, url],
                stdout=stream, stderr=subprocess.DEVNULL)
        if result.returncode == 0 and path.stat().st_size == expected:
            return
        if path.stat().st_size > expected:
            path.unlink(missing_ok=True)
        time.sleep(min(30, 2 ** min(attempt, 4)))
    raise RuntimeError(f"chunk {start}-{end} failed after {retries} attempts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.parent / (out.name + TEMP)
    temp.mkdir(exist_ok=True)

    size = head_size(args.url)
    bounds = []
    step = (size + args.parts - 1) // args.parts
    for i in range(args.parts):
        begin = i * step
        if begin >= size:
            break
        bounds.append((begin, min(begin + step, size) - 1))
    total = sum(e - s + 1 for s, e in bounds)
    print(f"size={size} parts={len(bounds)} total_to_fetch={total}", flush=True)

    started = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        jobs = []
        for index, (begin, end) in enumerate(bounds):
            jobs.append(pool.submit(
                fetch_chunk, args.url, begin, end,
                temp / f"part{index:04d}.bin"))
        done = 0
        for job in futures.as_completed(jobs):
            job.result()
            done += 1
            got = sum(p.stat().st_size for p in temp.glob("part*.bin"))
            rate = got / max(1e-9, time.perf_counter() - started) / 1e6
            print(f"chunks_done={done}/{len(jobs)} bytes={got} rate={rate:.2f}MB/s",
                  flush=True)

    with open(out, "wb") as stream:
        for index in range(len(bounds)):
            with open(temp / f"part{index:04d}.bin", "rb") as part:
                while True:
                    block = part.read(1 << 22)
                    if not block:
                        break
                    stream.write(block)
    actual = out.stat().st_size
    if actual != size:
        raise RuntimeError(f"size mismatch {actual} != {size}")
    print(f"written {out} bytes={actual}", flush=True)


if __name__ == "__main__":
    main()
