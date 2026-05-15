#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.libero_action_adapter import (  # noqa: E402
    rlds_open_to_libero_gripper,
    summarize_gripper_values,
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _get_parquet_paths(dataset_path: Path, info: dict) -> list[Path]:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    paths = []
    for episode_index in range(total_episodes):
        episode_chunk = episode_index // chunks_size
        path = dataset_path / pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if path.exists():
            paths.append(path)
    return paths


def _get_entry(modality: dict, section: str, key: str) -> dict:
    section_data = modality.get(section, {})
    if key in section_data:
        return section_data[key]
    prefixed = f"{section}.{key}"
    if prefixed in section_data:
        return section_data[prefixed]
    raise KeyError(f"Missing modality entry {section}.{key}. Available: {sorted(section_data)}")


def _read_values(
    parquet_paths: list[Path],
    entry: dict,
    max_rows: int | None,
    *,
    default_column: str,
) -> np.ndarray:
    column = entry.get("original_key", default_column)
    if not column:
        raise KeyError(f"modality entry has no original_key: {entry}")
    start = int(entry.get("start", 0))
    end = entry.get("end")
    end = None if end is None else int(end)

    chunks = []
    remaining = max_rows
    for path in parquet_paths:
        if remaining is not None and remaining <= 0:
            break
        df = pd.read_parquet(path, columns=[column])
        if remaining is not None:
            df = df.iloc[:remaining]
        values = df[column].to_numpy()
        if len(values) == 0:
            continue
        first = values[0]
        if np.isscalar(first):
            arr = values.astype(np.float32).reshape(-1, 1)
        else:
            arr = np.stack(values).astype(np.float32)
            arr = arr[..., start:end]
        chunks.append(arr.reshape(arr.shape[0], -1))
        if remaining is not None:
            remaining -= len(df)

    if not chunks:
        return np.empty((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _format_unique(values: np.ndarray, max_items: int = 12) -> str:
    rounded = np.round(values.reshape(-1), 6)
    unique, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(counts)[::-1]
    items = [f"{float(unique[i]):.6g}:{int(counts[i])}" for i in order[:max_items]]
    suffix = "" if len(unique) <= max_items else f" ... (+{len(unique) - max_items} more)"
    return ", ".join(items) + suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    info = _load_json(dataset_path / "meta" / "info.json")
    modality = _load_json(dataset_path / "meta" / "modality.json")
    parquet_paths = _get_parquet_paths(dataset_path, info)
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {dataset_path}")

    action_gripper_entry = _get_entry(modality, "action", "gripper")
    action_gripper = _read_values(
        parquet_paths,
        action_gripper_entry,
        args.max_rows,
        default_column="action",
    )
    if action_gripper.size == 0:
        raise SystemExit("No action gripper values found.")

    summary = summarize_gripper_values(action_gripper, threshold=args.threshold)
    print(f"dataset={dataset_path}")
    print(f"parquet_files={len(parquet_paths)} sampled_rows={summary['count']}")
    print(
        "action.gripper "
        f"min={summary['min']:.6g} max={summary['max']:.6g} mean={summary['mean']:.6g} "
        f"open_count={summary['open_count']} closed_count={summary['closed_count']} "
        f"open_ratio={summary['open_ratio']:.4f}"
    )
    print(f"action.gripper rounded_counts={_format_unique(action_gripper)}")

    env_gripper = rlds_open_to_libero_gripper(action_gripper, threshold=args.threshold, binarize=True)
    print(f"converted LIBERO env gripper rounded_counts={_format_unique(env_gripper)}")

    if np.min(action_gripper) < -0.05 or np.max(action_gripper) > 1.05:
        raise SystemExit(
            "action.gripper does not look like RLDS openness in [0, 1]. "
            "Check conversion before training/eval."
        )

    try:
        state_gripper_entry = _get_entry(modality, "state", "gripper")
    except KeyError:
        state_gripper_entry = None
    if state_gripper_entry is not None:
        state_gripper = _read_values(
            parquet_paths,
            state_gripper_entry,
            args.max_rows,
            default_column="observation.state",
        )
        if state_gripper.size:
            state_summary = summarize_gripper_values(state_gripper, threshold=args.threshold)
            print(
                "state.gripper "
                f"min={state_summary['min']:.6g} max={state_summary['max']:.6g} "
                f"mean={state_summary['mean']:.6g} open_ratio={state_summary['open_ratio']:.4f}"
            )


if __name__ == "__main__":
    main()
