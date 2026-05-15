from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "eval_utils" / "eval_libero_dreamzero.py"
DEFAULT_SUITES = "libero_spatial,libero_object,libero_goal,libero_10"
FASTWAM_NUM_STEPS_WAIT = 30


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_assignments(suites: list[str], num_workers: int) -> list[dict[str, list[int]]]:
    assignments = [defaultdict(list) for _ in range(num_workers)]
    flat_jobs = []
    for suite in suites:
        for task_id in range(10):
            flat_jobs.append((suite, task_id))
    for idx, (suite, task_id) in enumerate(flat_jobs):
        assignments[idx % num_workers][suite].append(task_id)
    return [dict(item) for item in assignments]


def _worker(args: argparse.Namespace) -> int:
    suites = _parse_csv(args.suites)
    gpus = [int(gpu) for gpu in _parse_csv(args.gpus)]
    assignments = _build_assignments(suites, len(gpus))
    worker_id = args.worker_id
    gpu_id = gpus[worker_id]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"worker{worker_id}_gpu{gpu_id}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MASTER_PORT"] = str(args.master_port + worker_id)

    with log_path.open("a", encoding="utf-8") as log:
        print(f"[worker {worker_id}] gpu={gpu_id} assignments={assignments[worker_id]}", file=log, flush=True)
        for suite in suites:
            task_ids = assignments[worker_id].get(suite, [])
            if not task_ids:
                continue
            result_json = output_dir / f"worker{worker_id}_{suite}.json"
            task_result_dir = output_dir / suite
            cmd = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--model-path",
                args.model_path,
                "--task-suite",
                suite,
                "--task-ids",
                ",".join(str(task_id) for task_id in task_ids),
                "--num-trials",
                str(args.num_trials),
                "--seed",
                str(args.seed),
                "--gpu-id",
                str(gpu_id),
                "--replan-steps",
                str(args.replan_steps),
                "--num-steps-wait",
                str(args.num_steps_wait),
                "--num-envs",
                str(args.num_envs),
                "--result-json",
                str(result_json),
                "--task-result-dir",
                str(task_result_dir),
                "--gripper-threshold",
                str(args.gripper_threshold),
            ]
            if args.no_binarize_gripper:
                cmd.append("--no-binarize-gripper")
            if args.parallel_env_step:
                cmd.append("--parallel-env-step")
            if args.subproc_env_step:
                cmd.append("--subproc-env-step")
            if args.max_steps is not None:
                cmd.extend(["--max-steps", str(args.max_steps)])
            print(f"[worker {worker_id}] start {' '.join(cmd)}", file=log, flush=True)
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
            print(
                f"[worker {worker_id}] finished suite={suite} returncode={completed.returncode}",
                file=log,
                flush=True,
            )
            if completed.returncode != 0:
                return completed.returncode
    return 0


def _launch_detached(args: argparse.Namespace) -> None:
    suites = _parse_csv(args.suites)
    gpus = [int(gpu) for gpu in _parse_csv(args.gpus)]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = _build_assignments(suites, len(gpus))
    launch = {
        "model_path": args.model_path,
        "suites": suites,
        "gpus": gpus,
        "num_trials": args.num_trials,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "max_steps": args.max_steps,
        "num_envs": args.num_envs,
        "parallel_env_step": args.parallel_env_step,
        "subproc_env_step": args.subproc_env_step,
        "gripper_threshold": args.gripper_threshold,
        "no_binarize_gripper": args.no_binarize_gripper,
        "assignments": assignments,
        "workers": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for worker_id, gpu_id in enumerate(gpus):
        log_path = output_dir / f"worker{worker_id}_gpu{gpu_id}.log"
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--model-path",
            args.model_path,
            "--suites",
            args.suites,
            "--gpus",
            args.gpus,
            "--num-trials",
            str(args.num_trials),
            "--seed",
            str(args.seed),
            "--replan-steps",
            str(args.replan_steps),
            "--num-steps-wait",
            str(args.num_steps_wait),
            "--num-envs",
            str(args.num_envs),
            "--master-port",
            str(args.master_port),
            "--output-dir",
            str(output_dir),
            "--worker-id",
            str(worker_id),
            "--gripper-threshold",
            str(args.gripper_threshold),
        ]
        if args.no_binarize_gripper:
            cmd.append("--no-binarize-gripper")
        if args.parallel_env_step:
            cmd.append("--parallel-env-step")
        if args.subproc_env_step:
            cmd.append("--subproc-env-step")
        if args.max_steps is not None:
            cmd.extend(["--max-steps", str(args.max_steps)])
        log_file = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        log_file.close()
        launch["workers"].append({"worker_id": worker_id, "gpu_id": gpu_id, "pid": process.pid, "log": str(log_path)})

    launch_path = output_dir / "launch.json"
    launch_path.write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(f"launched {len(gpus)} workers")
    print(f"output_dir={output_dir}")
    for worker in launch["workers"]:
        print(f"worker{worker['worker_id']} gpu={worker['gpu_id']} pid={worker['pid']} log={worker['log']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--suites", default=DEFAULT_SUITES)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=FASTWAM_NUM_STEPS_WAIT)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of LIBERO envs to run in parallel per model process.")
    parser.add_argument("--parallel-env-step", action="store_true", help="Step vectorized LIBERO environments concurrently with a thread pool.")
    parser.add_argument("--subproc-env-step", action="store_true", help="Step vectorized LIBERO environments in separate subprocesses.")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--master-port", type=int, default=29543)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker-id", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "eval_results" / f"dreamzero_libero_standard_{stamp}")

    if args.worker_id is not None:
        raise SystemExit(_worker(args))
    if args.detach:
        _launch_detached(args)
        return
    raise SystemExit("Use --detach for background two-GPU evaluation or --worker-id for an internal worker.")


if __name__ == "__main__":
    main()
