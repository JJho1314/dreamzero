from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tianshou.data import Batch

_DYNAMO = torch._dynamo.config
if hasattr(_DYNAMO, "cache_size_limit"):
    _DYNAMO.cache_size_limit = 1000
if hasattr(_DYNAMO, "recompile_limit"):
    _DYNAMO.recompile_limit = 800
if hasattr(_DYNAMO, "accumulated_cache_size_limit"):
    _DYNAMO.accumulated_cache_size_limit = 1000
if hasattr(_DYNAMO, "accumulated_recompile_limit"):
    _DYNAMO.accumulated_recompile_limit = 2000


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIBERO_ROOT = pathlib.Path(os.environ.get("LIBERO_HOME", "/data/LFT-W02_data/junjie/LIBERO"))
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))
ROBOSUITE_SITE = pathlib.Path(
    os.environ.get(
        "ROBOSUITE_SITE",
        "/data/LFT-W02_data/.conda/envs/fastwam/lib/python3.10/site-packages",
    )
)
if ROBOSUITE_SITE.exists() and str(ROBOSUITE_SITE) not in sys.path:
    sys.path.append(str(ROBOSUITE_SITE))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "libero"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("DISABLE_DREAMZERO_TORCH_COMPILE", "true")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# The LIBERO eval imports robosuite from the FastWAM py3.10 environment while
# running under the DreamZero py3.11 environment. Robosuite asks numba to cache
# jitted functions from that foreign site-packages path, which can fail before
# argparse even runs. Disable numba's disk cache in-process for this eval entry.
try:
    import numba as _numba

    _ORIGINAL_NUMBA_JIT = _numba.jit

    def _numba_jit_no_disk_cache(*args, **kwargs):
        kwargs["cache"] = False
        return _ORIGINAL_NUMBA_JIT(*args, **kwargs)

    _numba.jit = _numba_jit_no_disk_cache
except Exception:
    pass

_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402
from eval_utils.libero_action_adapter import rlds_open_to_libero_gripper  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


LOG = logging.getLogger("dreamzero_libero_eval")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_MODEL_VIEW_SIZE = int(os.getenv("DREAMZERO_LIBERO_VIEW_SIZE", "224"))
FRAMES_PER_CHUNK = 4
FASTWAM_NUM_STEPS_WAIT = 30


@dataclass
class EvalArgs:
    model_path: str
    task_suite: str = "libero_goal"
    task_id: int = 0
    num_trials: int = 1
    seed: int = 7
    gpu_id: int = 0
    replan_steps: int = 10
    num_steps_wait: int = FASTWAM_NUM_STEPS_WAIT
    max_steps: Optional[int] = None
    num_envs: int = 1
    parallel_env_step: bool = False
    subproc_env_step: bool = False
    video_out: Optional[str] = None
    tokenizer_path: str = "/data/LFT-W02_data/junjie/Wan2.2/weights/TI2V_5B/google/umt5-xxl"
    task_ids: Optional[str] = None
    all_tasks: bool = False
    result_json: Optional[str] = None
    task_result_dir: Optional[str] = None
    gripper_threshold: float = 0.5
    no_binarize_gripper: bool = False


def _maybe_init_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29543")
    dist.init_process_group("nccl", rank=0, world_size=1)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _center_crop_resize(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(image)
    src_w, src_h = pil.size
    scale = max(size / src_w, size / src_h)
    resized = pil.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - size) // 2, 0)
    top = max((rh - size) // 2, 0)
    return np.asarray(resized.crop((left, top, left + size, top + size)), dtype=np.uint8)


def _get_libero_images(obs: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return _center_crop_resize(primary, size), _center_crop_resize(wrist, size)


def _state_from_obs(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    eef_pose = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(obs["robot0_eef_quat"]).reshape(-1),
        )
    )
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper_qpos.size == 0:
        raise ValueError("LIBERO observation is missing robot0_gripper_qpos values")
    if gripper_qpos.size == 1:
        gripper_qpos = np.repeat(gripper_qpos, 2)
    return eef_pose.astype(np.float64), gripper_qpos[:2].astype(np.float64)


def _infer_model_view_size(policy: GrootSimPolicy) -> int:
    if "DREAMZERO_LIBERO_VIEW_SIZE" in os.environ:
        return DEFAULT_MODEL_VIEW_SIZE
    try:
        action_head = policy.trained_model.action_head
        cfg = action_head.config
        target_h = int(getattr(cfg, "target_video_height", 0) or 0)
        target_w = int(getattr(cfg, "target_video_width", 0) or 0)
    except Exception:
        return DEFAULT_MODEL_VIEW_SIZE
    if target_h > 0 and target_w == 2 * target_h:
        return target_h
    return DEFAULT_MODEL_VIEW_SIZE


def _gripper_state_keys(gripper_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Converted LIBERO metadata names the two qpos scalars state.pad and state.gripper.
    return gripper_qpos[:1], gripper_qpos[1:2]


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _extract_action_chunk(result_batch: Batch) -> np.ndarray:
    return _extract_action_chunks(result_batch)[0]


def _extract_action_chunks(result_batch: Batch) -> np.ndarray:
    act = result_batch.act
    action_dict = {}
    for key in dir(act):
        if key.startswith("action."):
            action_dict[key] = getattr(act, key)
    if "action.eef" not in action_dict or "action.gripper" not in action_dict:
        raise KeyError(f"Missing DreamZero action keys. Found: {sorted(action_dict)}")
    eef = _to_numpy(action_dict["action.eef"])
    gripper = _to_numpy(action_dict["action.gripper"])
    if eef.ndim == 1:
        eef = eef.reshape(1, 1, -1)
    elif eef.ndim == 2:
        eef = eef[None, ...]
    elif eef.ndim != 3:
        raise ValueError(f"Unexpected DreamZero eef action shape: {eef.shape}")

    if gripper.ndim == 1:
        gripper = gripper.reshape(1, -1, 1)
    elif gripper.ndim == 2:
        if gripper.shape[0] == eef.shape[0] and gripper.shape[1] == eef.shape[1]:
            gripper = gripper[..., None]
        else:
            gripper = gripper[None, ...]
    elif gripper.ndim != 3:
        raise ValueError(f"Unexpected DreamZero gripper action shape: {gripper.shape}")
    if eef.ndim != 3 or gripper.ndim != 3:
        raise ValueError(f"Unexpected DreamZero action shapes: eef={eef.shape}, gripper={gripper.shape}")
    if eef.shape[:2] != gripper.shape[:2]:
        raise ValueError(f"DreamZero action length mismatch: eef={eef.shape}, gripper={gripper.shape}")
    return np.concatenate([eef, gripper], axis=-1).astype(np.float32)


def _env_gripper_from_rlds(
    open_value: np.ndarray,
    *,
    threshold: float = 0.5,
    binarize: bool = True,
) -> np.ndarray:
    return rlds_open_to_libero_gripper(open_value, threshold=threshold, binarize=binarize).reshape(1)


class DreamZeroLiberoPolicy:
    def __init__(self, policy: GrootSimPolicy):
        self.policy = policy
        self.view_size = _infer_model_view_size(policy)
        self.primary_frames: deque[np.ndarray] = deque(maxlen=FRAMES_PER_CHUNK)
        self.wrist_frames: deque[np.ndarray] = deque(maxlen=FRAMES_PER_CHUNK)
        self.first_call = True

    def reset(self) -> None:
        self.primary_frames.clear()
        self.wrist_frames.clear()
        self.first_call = True
        action_head = self.policy.trained_model.action_head
        action_head.current_start_frame = 0
        action_head.language = None
        action_head.kv_cache1 = None
        action_head.kv_cache_neg = None
        action_head.crossattn_cache = None
        action_head.crossattn_cache_neg = None

    def predict_action_chunk(self, obs: dict, task_description: str) -> np.ndarray:
        primary, wrist = _get_libero_images(obs, self.view_size)
        self.primary_frames.append(primary)
        self.wrist_frames.append(wrist)

        num_frames = 1 if self.first_call else FRAMES_PER_CHUNK
        primary_frames = list(self.primary_frames)
        wrist_frames = list(self.wrist_frames)
        while len(primary_frames) < num_frames:
            primary_frames.insert(0, primary_frames[0])
            wrist_frames.insert(0, wrist_frames[0])

        eef_pose, gripper_qpos = _state_from_obs(obs)
        gripper_pad, gripper = _gripper_state_keys(gripper_qpos)
        converted = {
            "video.primary_image": np.stack(primary_frames[-num_frames:], axis=0),
            "video.wrist_image": np.stack(wrist_frames[-num_frames:], axis=0),
            "state.eef_pose": eef_pose.reshape(1, -1),
            "state.pad": gripper_pad.reshape(1, -1),
            "state.gripper": gripper.reshape(1, -1),
            "annotation.task": str(task_description),
        }
        with torch.no_grad():
            result_batch, _ = self.policy.lazy_joint_forward_causal(Batch(obs=converted))
        self.first_call = False
        return _extract_action_chunk(result_batch)


class DreamZeroLiberoVectorPolicy:
    def __init__(self, policy: GrootSimPolicy):
        self.policy = policy
        self.view_size = _infer_model_view_size(policy)
        self.primary_frames: list[deque[np.ndarray]] = []
        self.wrist_frames: list[deque[np.ndarray]] = []
        self.first_calls: list[bool] = []

    def reset(self, num_envs: int) -> None:
        self.primary_frames = [deque(maxlen=FRAMES_PER_CHUNK) for _ in range(num_envs)]
        self.wrist_frames = [deque(maxlen=FRAMES_PER_CHUNK) for _ in range(num_envs)]
        self.first_calls = [True for _ in range(num_envs)]
        action_head = self.policy.trained_model.action_head
        action_head.current_start_frame = 0
        action_head.language = None
        action_head.kv_cache1 = None
        action_head.kv_cache_neg = None
        action_head.crossattn_cache = None
        action_head.crossattn_cache_neg = None

    def predict_action_chunks(self, obses: list[dict], task_description: str) -> np.ndarray:
        if not obses:
            raise ValueError("predict_action_chunks requires at least one observation")
        if len(obses) != len(self.primary_frames):
            raise ValueError(f"Expected {len(self.primary_frames)} env observations, got {len(obses)}")

        primary_batch = []
        wrist_batch = []
        eef_batch = []
        gripper_pad_batch = []
        gripper_batch = []
        num_frames = 1 if all(self.first_calls) else FRAMES_PER_CHUNK

        for env_index, obs in enumerate(obses):
            primary, wrist = _get_libero_images(obs, self.view_size)
            self.primary_frames[env_index].append(primary)
            self.wrist_frames[env_index].append(wrist)

            primary_frames = list(self.primary_frames[env_index])
            wrist_frames = list(self.wrist_frames[env_index])
            while len(primary_frames) < num_frames:
                primary_frames.insert(0, primary_frames[0])
                wrist_frames.insert(0, wrist_frames[0])

            eef_pose, gripper_qpos = _state_from_obs(obs)
            gripper_pad, gripper = _gripper_state_keys(gripper_qpos)
            primary_batch.append(np.stack(primary_frames[-num_frames:], axis=0))
            wrist_batch.append(np.stack(wrist_frames[-num_frames:], axis=0))
            eef_batch.append(eef_pose.reshape(1, -1))
            gripper_pad_batch.append(gripper_pad.reshape(1, -1))
            gripper_batch.append(gripper.reshape(1, -1))

        converted = {
            "video.primary_image": np.stack(primary_batch, axis=0),
            "video.wrist_image": np.stack(wrist_batch, axis=0),
            "state.eef_pose": np.stack(eef_batch, axis=0),
            "state.pad": np.stack(gripper_pad_batch, axis=0),
            "state.gripper": np.stack(gripper_batch, axis=0),
            "annotation.task": np.asarray([str(task_description)] * len(obses)),
        }
        with torch.no_grad():
            result_batch, _ = self.policy.lazy_joint_forward_causal(Batch(obs=converted))
        self.first_calls = [False for _ in self.first_calls]
        return _extract_action_chunks(result_batch)


def _default_max_steps(task_suite: str) -> int:
    return {
        "libero_spatial": 800,
        "libero_object": 800,
        "libero_goal": 800,
        "libero_10": 800,
        "libero_90": 800,
    }[task_suite]


def _parse_task_ids(task_ids: str) -> list[int]:
    parsed = []
    for part in task_ids.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            parsed.extend(range(int(start), int(end) + 1))
        else:
            parsed.append(int(item))
    return parsed


def _resolve_task_ids(args: EvalArgs, task_suite) -> list[int]:
    if args.all_tasks:
        return list(range(task_suite.n_tasks))
    if args.task_ids:
        return _parse_task_ids(args.task_ids)
    return [args.task_id]


def _write_task_result(args: EvalArgs, task_result: dict) -> None:
    if not args.task_result_dir:
        return
    out_dir = pathlib.Path(args.task_result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gpu{args.gpu_id}_task{task_result['task_id']}_results.json"
    payload = {
        "model_path": args.model_path,
        "task_suite": args.task_suite,
        "task_id": task_result["task_id"],
        "task_description": task_result["task"],
        "successes": task_result["successes"],
        "total_episodes": task_result["trials"],
        "success_rate": task_result["success_rate"],
        "gpu_id": args.gpu_id,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "max_steps": task_result["max_steps"],
        "num_envs": args.num_envs,
        "parallel_env_step": args.parallel_env_step,
        "subproc_env_step": args.subproc_env_step,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("wrote task result json: %s", out_path)


def _subproc_env_worker(remote, bddl_file_name: str, seed: int) -> None:
    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(seed)
        remote.send(("ready", None))
        while True:
            cmd, payload = remote.recv()
            if cmd == "close":
                remote.send(("ok", None))
                break
            if cmd == "reset_to_state":
                env.reset()
                remote.send(("ok", env.set_init_state(payload)))
                continue
            if cmd == "step":
                obs, _, done, _ = env.step(payload)
                remote.send(("ok", (obs, done)))
                continue
            raise ValueError(f"Unknown env worker command: {cmd}")
    except BaseException as exc:
        try:
            remote.send(("error", repr(exc)))
        except BaseException:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        remote.close()


class SubprocLiberoEnvPool:
    def __init__(self, bddl_file_name: str, num_envs: int, seed: int):
        # Use spawn so env workers do not inherit the CUDA context from the model process.
        self.ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        for env_index in range(num_envs):
            parent_remote, child_remote = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_subproc_env_worker,
                args=(child_remote, bddl_file_name, seed + env_index),
            )
            process.daemon = True
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)
        for remote in self.remotes:
            self._recv(remote)

    @staticmethod
    def _recv(remote):
        status, payload = remote.recv()
        if status != "ok" and status != "ready":
            raise RuntimeError(f"LIBERO env worker failed: {payload}")
        return payload

    def reset_to_states(self, states: list[np.ndarray]) -> list[dict]:
        for env_index, state in enumerate(states):
            self.remotes[env_index].send(("reset_to_state", state))
        return [self._recv(self.remotes[env_index]) for env_index in range(len(states))]

    def step(self, items: list[tuple[int, list[float]]]) -> list[tuple[int, dict, bool]]:
        for env_index, action in items:
            self.remotes[env_index].send(("step", action))
        return [
            (env_index, *self._recv(self.remotes[env_index]))
            for env_index, _ in items
        ]

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for remote in self.remotes:
            try:
                self._recv(remote)
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            remote.close()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()


def _run_one_task(args: EvalArgs, policy: DreamZeroLiberoPolicy, task_suite, task_id: int) -> dict:
    task = task_suite.get_task(task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(task_id)
    if args.num_trials > len(initial_states):
        raise ValueError(
            f"Requested {args.num_trials} trials but only {len(initial_states)} init states "
            f"are available for {args.task_suite} task {task_id}."
        )
    max_steps = args.max_steps or _default_max_steps(args.task_suite)

    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)
    if args.video_out:
        pathlib.Path(args.video_out).mkdir(parents=True, exist_ok=True)

    successes = 0
    LOG.info(
        "suite=%s task_id=%d/%d trials=%d max_steps=%d task=%s",
        args.task_suite,
        task_id,
        task_suite.n_tasks,
        args.num_trials,
        max_steps,
        task_description,
    )

    for trial in range(args.num_trials):
        policy.reset()
        env.reset()
        obs = env.set_init_state(initial_states[trial])
        done = False
        replay = []
        pending_actions: list[np.ndarray] = []

        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                continue
            if args.video_out:
                replay.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if not pending_actions:
                chunk = policy.predict_action_chunk(obs, task_description)
                keep = min(args.replan_steps, len(chunk))
                pending_actions = [chunk[i] for i in range(keep)]
            action = pending_actions.pop(0)
            env_action = np.concatenate(
                [
                    action[:6],
                    _env_gripper_from_rlds(
                        action[6:7],
                        threshold=args.gripper_threshold,
                        binarize=not args.no_binarize_gripper,
                    ),
                ],
                axis=0,
            )
            obs, _, done, _ = env.step(env_action.tolist())
            if done:
                successes += 1
                break

        LOG.info("trial %d/%d: %s", trial + 1, args.num_trials, "SUCCESS" if done else "fail")
        if args.video_out and replay:
            import imageio

            status = "success" if done else "fail"
            out_path = pathlib.Path(args.video_out) / f"{args.task_suite}_task{task_id}_trial{trial}_{status}.mp4"
            imageio.mimwrite(str(out_path), replay, fps=10)
            LOG.info("saved video: %s", out_path)

    env.close()
    success_rate = 100.0 * successes / max(args.num_trials, 1)
    LOG.info("TASK FINAL SR: %s task %d: %d/%d = %.1f%%", args.task_suite, task_id, successes, args.num_trials, success_rate)
    return {
        "suite": args.task_suite,
        "task_id": task_id,
        "task": task_description,
        "successes": successes,
        "trials": args.num_trials,
        "success_rate": success_rate,
        "max_steps": max_steps,
    }


def _run_one_task_vectorized(args: EvalArgs, policy: DreamZeroLiberoVectorPolicy, task_suite, task_id: int) -> dict:
    task = task_suite.get_task(task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(task_id)
    if args.num_trials > len(initial_states):
        raise ValueError(
            f"Requested {args.num_trials} trials but only {len(initial_states)} init states "
            f"are available for {args.task_suite} task {task_id}."
        )
    max_steps = args.max_steps or _default_max_steps(args.task_suite)
    num_envs = max(1, int(args.num_envs))
    use_subproc_env_step = args.subproc_env_step and num_envs > 1
    use_parallel_env_step = args.parallel_env_step and num_envs > 1 and not use_subproc_env_step

    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_pool = SubprocLiberoEnvPool(str(bddl_path), num_envs, args.seed) if use_subproc_env_step else None
    envs = []
    if env_pool is None:
        envs = [
            OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            for _ in range(num_envs)
        ]
        for env in envs:
            env.seed(args.seed)
    if args.video_out:
        pathlib.Path(args.video_out).mkdir(parents=True, exist_ok=True)

    successes = 0
    LOG.info(
        "suite=%s task_id=%d/%d trials=%d max_steps=%d num_envs=%d parallel_env_step=%s subproc_env_step=%s task=%s",
        args.task_suite,
        task_id,
        task_suite.n_tasks,
        args.num_trials,
        max_steps,
        num_envs,
        use_parallel_env_step,
        use_subproc_env_step,
        task_description,
    )

    try:
        executor = ThreadPoolExecutor(max_workers=num_envs) if use_parallel_env_step else None

        def reset_to_state(item):
            env, state = item
            env.reset()
            return env.set_init_state(state)

        def step_with_action(item):
            env_index, env, action = item
            obs, _, done, _ = env.step(action)
            return env_index, obs, done

        for group_start in range(0, args.num_trials, num_envs):
            group_trials = list(range(group_start, min(group_start + num_envs, args.num_trials)))
            batch_size = len(group_trials)
            policy.reset(batch_size)

            obses = []
            dones = [False for _ in group_trials]
            replays: list[list[np.ndarray]] = [[] for _ in group_trials]
            pending_actions: list[list[np.ndarray]] = [[] for _ in group_trials]

            if env_pool is not None:
                obses = env_pool.reset_to_states([initial_states[trial] for trial in group_trials])
            else:
                active_envs = envs[:batch_size]
                reset_items = [(active_envs[env_index], initial_states[trial]) for env_index, trial in enumerate(group_trials)]
                if executor is not None:
                    obses = list(executor.map(reset_to_state, reset_items))
                else:
                    obses = [reset_to_state(item) for item in reset_items]

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    if env_pool is not None:
                        step_results = env_pool.step([
                            (env_index, LIBERO_DUMMY_ACTION)
                            for env_index in range(batch_size)
                            if not dones[env_index]
                        ])
                    else:
                        step_items = [
                            (env_index, env, LIBERO_DUMMY_ACTION)
                            for env_index, env in enumerate(active_envs)
                            if not dones[env_index]
                        ]
                        step_results = executor.map(step_with_action, step_items) if executor is not None else map(step_with_action, step_items)
                    for env_index, obs, _ in step_results:
                        obses[env_index] = obs
                    continue

                if args.video_out:
                    for env_index, obs in enumerate(obses):
                        if not dones[env_index]:
                            replays[env_index].append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

                if any(not done and not pending_actions[env_index] for env_index, done in enumerate(dones)):
                    chunks = policy.predict_action_chunks(obses, task_description)
                    for env_index, done in enumerate(dones):
                        if done:
                            continue
                        keep = min(args.replan_steps, chunks.shape[1])
                        pending_actions[env_index] = [chunks[env_index, i] for i in range(keep)]

                step_items = []
                for env_index in range(batch_size):
                    if dones[env_index]:
                        continue
                    action = pending_actions[env_index].pop(0)
                    env_action = np.concatenate(
                        [
                            action[:6],
                            _env_gripper_from_rlds(
                                action[6:7],
                                threshold=args.gripper_threshold,
                                binarize=not args.no_binarize_gripper,
                            ),
                        ],
                        axis=0,
                    )
                    if env_pool is not None:
                        step_items.append((env_index, env_action.tolist()))
                    else:
                        step_items.append((env_index, active_envs[env_index], env_action.tolist()))
                if env_pool is not None:
                    step_results = env_pool.step(step_items)
                else:
                    step_results = executor.map(step_with_action, step_items) if executor is not None else map(step_with_action, step_items)
                for env_index, obs, done in step_results:
                    obses[env_index] = obs
                    dones[env_index] = done
                    if done:
                        successes += 1

                if all(dones):
                    break

            for env_index, trial in enumerate(group_trials):
                done = dones[env_index]
                LOG.info("trial %d/%d: %s", trial + 1, args.num_trials, "SUCCESS" if done else "fail")
                if args.video_out and replays[env_index]:
                    import imageio

                    status = "success" if done else "fail"
                    out_path = pathlib.Path(args.video_out) / f"{args.task_suite}_task{task_id}_trial{trial}_{status}.mp4"
                    imageio.mimwrite(str(out_path), replays[env_index], fps=10)
                    LOG.info("saved video: %s", out_path)
    finally:
        if "executor" in locals() and executor is not None:
            executor.shutdown(wait=True)
        if env_pool is not None:
            env_pool.close()
        for env in envs:
            env.close()

    success_rate = 100.0 * successes / max(args.num_trials, 1)
    LOG.info("TASK FINAL SR: %s task %d: %d/%d = %.1f%%", args.task_suite, task_id, successes, args.num_trials, success_rate)
    return {
        "suite": args.task_suite,
        "task_id": task_id,
        "task": task_description,
        "successes": successes,
        "trials": args.num_trials,
        "success_rate": success_rate,
        "max_steps": max_steps,
    }


def run(args: EvalArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)
    _maybe_init_dist()

    LOG.info("loading DreamZero checkpoint: %s", args.model_path)
    groot_policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device="cuda",
        tokenizer_path_override=args.tokenizer_path,
        skip_img_transform=True,
    )
    for transform in getattr(groot_policy.eval_transform, "transforms", []):
        if hasattr(transform, "max_chunk_size"):
            transform.max_chunk_size = 1
        if hasattr(transform, "num_frames"):
            transform.num_frames = None
    policy = DreamZeroLiberoPolicy(groot_policy)
    vector_policy = DreamZeroLiberoVectorPolicy(groot_policy)
    LOG.info("model loaded; cuda memory %.2f GB", torch.cuda.max_memory_allocated() / 1024**3)

    bench_cls = benchmark.get_benchmark(args.task_suite)
    task_suite = bench_cls()
    task_ids = _resolve_task_ids(args, task_suite)

    results = []
    for task_id in task_ids:
        if args.num_envs > 1:
            task_result = _run_one_task_vectorized(args, vector_policy, task_suite, task_id)
        else:
            task_result = _run_one_task(args, policy, task_suite, task_id)
        results.append(task_result)
        _write_task_result(args, task_result)

    total_successes = sum(item["successes"] for item in results)
    total_trials = sum(item["trials"] for item in results)
    LOG.info(
        "SUITE FINAL SR: %s tasks=%s: %d/%d = %.1f%%",
        args.task_suite,
        task_ids,
        total_successes,
        total_trials,
        100.0 * total_successes / max(total_trials, 1),
    )

    if args.result_json:
        out_path = pathlib.Path(args.result_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "suite": args.task_suite,
                    "task_ids": task_ids,
                    "num_trials": args.num_trials,
                    "num_envs": args.num_envs,
                    "parallel_env_step": args.parallel_env_step,
                    "subproc_env_step": args.subproc_env_step,
                    "successes": total_successes,
                    "trials": total_trials,
                    "success_rate": 100.0 * total_successes / max(total_trials, 1),
                    "tasks": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOG.info("wrote result json: %s", out_path)

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task-suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=FASTWAM_NUM_STEPS_WAIT)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of LIBERO envs to run in parallel inside one model process.")
    parser.add_argument("--parallel-env-step", action="store_true", help="Step vectorized LIBERO environments concurrently with a thread pool.")
    parser.add_argument("--subproc-env-step", action="store_true", help="Step vectorized LIBERO environments in separate subprocesses.")
    parser.add_argument("--video-out", default=None)
    parser.add_argument("--tokenizer-path", default="/data/LFT-W02_data/junjie/Wan2.2/weights/TI2V_5B/google/umt5-xxl")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids or ranges, e.g. 0,2,4-7")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--task-result-dir", default=None)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    ns = parser.parse_args()
    run(EvalArgs(**vars(ns)))


if __name__ == "__main__":
    main()
