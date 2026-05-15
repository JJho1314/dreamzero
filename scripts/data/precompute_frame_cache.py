#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import decord
import numpy as np


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    resized = np.empty((frames.shape[0], size, size, frames.shape[3]), dtype=np.uint8)
    for i, frame in enumerate(frames):
        resized[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def _convert_one(args: tuple[str, str, str, int, bool]) -> tuple[str, str, int, int]:
    src_str, source_root_str, cache_root_str, size, overwrite = args
    src = Path(src_str)
    source_root = Path(source_root_str)
    cache_root = Path(cache_root_str)
    dst = cache_root / src.relative_to(source_root).with_suffix(".npy")

    if dst.exists() and not overwrite:
        arr = np.load(dst, mmap_mode="r")
        return ("skip", src.as_posix(), int(arr.shape[0]), int(dst.stat().st_size))

    dst.parent.mkdir(parents=True, exist_ok=True)
    vr = decord.VideoReader(src.as_posix())
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    frames = _resize_frames(frames, size)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, frames)
    tmp.replace(dst)
    return ("write", src.as_posix(), int(frames.shape[0]), int(dst.stat().st_size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    videos = sorted(source_root.rglob("*.mp4"))
    print(f"source_root={source_root}")
    print(f"cache_root={cache_root}")
    print(f"videos={len(videos)} size={args.size} workers={args.workers}")

    total_frames = 0
    total_bytes = 0
    failures: list[tuple[str, str]] = []
    tasks = [
        (p.as_posix(), source_root.as_posix(), cache_root.as_posix(), args.size, args.overwrite)
        for p in videos
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {executor.submit(_convert_one, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(future_to_path), start=1):
            path = future_to_path[future]
            try:
                status, src, frames, nbytes = future.result()
                total_frames += frames
                total_bytes += nbytes
                if i % 50 == 0 or status == "write":
                    print(
                        f"[{i}/{len(videos)}] {status} frames={frames} "
                        f"cache_mb={nbytes / 1024 / 1024:.2f} {src}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append((path, repr(exc)))
                print(f"[{i}/{len(videos)}] FAIL {path} {exc!r}", flush=True)

    print(
        f"done videos={len(videos)} frames={total_frames} "
        f"cache_gb={total_bytes / 1024 / 1024 / 1024:.2f} failures={len(failures)}",
        flush=True,
    )
    for path, exc in failures:
        print(f"FAILURE_DETAIL {path} {exc}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
