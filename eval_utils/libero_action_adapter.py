from __future__ import annotations

import numpy as np


def rlds_open_to_libero_gripper(
    open_value: np.ndarray | float,
    *,
    threshold: float = 0.5,
    binarize: bool = True,
) -> np.ndarray:
    """Convert RLDS gripper openness to LIBERO env gripper command.

    RLDS-style LIBERO data stores gripper as openness: 0=closed, 1=open.
    LIBERO/robosuite env actions use the opposite signed command:
    -1=open, +1=close.
    """
    value = np.asarray(open_value, dtype=np.float32)
    if binarize:
        value = (value > threshold).astype(np.float32)
    else:
        value = np.clip(value, 0.0, 1.0)
    return (1.0 - 2.0 * value).astype(np.float32)


def libero_env_action_from_rlds(
    action: np.ndarray,
    *,
    threshold: float = 0.5,
    binarize_gripper: bool = True,
) -> np.ndarray:
    """Convert a 7D RLDS action [eef(6), gripper_open] to LIBERO env action."""
    env_action = np.asarray(action, dtype=np.float32).copy()
    if env_action.shape[-1] < 7:
        raise ValueError(f"Expected action last dim >= 7, got shape {env_action.shape}")
    env_action[..., -1:] = rlds_open_to_libero_gripper(
        env_action[..., -1:],
        threshold=threshold,
        binarize=binarize_gripper,
    )
    return env_action


def summarize_gripper_values(values: np.ndarray, *, threshold: float = 0.5) -> dict[str, float | int]:
    gripper = np.asarray(values, dtype=np.float32).reshape(-1)
    if gripper.size == 0:
        raise ValueError("No gripper values to summarize.")
    return {
        "count": int(gripper.size),
        "min": float(np.min(gripper)),
        "max": float(np.max(gripper)),
        "mean": float(np.mean(gripper)),
        "open_count": int(np.sum(gripper > threshold)),
        "closed_count": int(np.sum(gripper <= threshold)),
        "open_ratio": float(np.mean(gripper > threshold)),
    }
