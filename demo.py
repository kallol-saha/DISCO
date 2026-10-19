"""
Demo: execute DISCO policy-value rollouts and report metrics.
"""
import argparse
import os

import numpy as np
import torch

from disco.tasks import get_task

TASK_DIR = {"shelf": "shelf_packing", "stacking": "object_stacking"}
ENV_ROOT = "assets/environments/{task}/{split}"
ROLLOUTS = "assets/rollouts/{task}/{split}/env_{env}"


def count(task, env):
    """Return slot 0's object count: objects in the shelf (shelf) or tallest tower height (stacking).

    Args:
        task: Task adapter from disco.tasks.
        env: Simulator holding the state to count.
    """
    if task.name == "shelf":
        return sum(env.check_object_inside_shelf(name, env_idx=0) for name in env.object_names)
    return int(env.tallest_tower(0)[0])


def set_all(env, state):
    """Set every sim slot to the same state.

    Args:
        env: 32-slot simulator.
        state: Flat state vector of one slot.
    """
    env.set_state(torch.as_tensor(state).float().to(env.device).unsqueeze(0).repeat(env.num_envs, 1))


def run_env(task, env_root, env_index, rollout_dir):
    """Execute a DISCO policy-value rollout.

    Args:
        task: Task adapter from disco.tasks.
        env_root: Directory of env scenes (env_<i>/).
        env_index: Env to load.
        rollout_dir: Rollout folder with initial_state.pt and step_<k>.npz files.

    Returns:
        Final object count (see `count`).
    """
    env = task.make_env(env_root, env_index)
    state, k = torch.load(f"{rollout_dir}/initial_state.pt"), 0
    while os.path.exists(f"{rollout_dir}/step_{k}.npz"):
        step = np.load(f"{rollout_dir}/step_{k}.npz")
        set_all(env, state)
        env.robot_controller.execute_trajectory(np.repeat(step["trajectory"][None], env.num_envs, axis=0))
        if task.name == "stacking":
            env.wait_for_stability(max_steps=200)
        state, k = step["state"], k + 1
    set_all(env, state)
    final = count(task, env)
    env.close()
    return final


def main():
    """Execute DISCO policy-value rollouts and print the success rate."""
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument("--task", choices=["shelf", "stacking"], required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--env", type=int, default=None, help="one env only")
    args = ap.parse_args()

    task = get_task(args.task)
    env_root = ENV_ROOT.format(task=TASK_DIR[args.task], split=args.split)
    envs = [args.env] if args.env is not None else sorted(int(d[4:]) for d in os.listdir(env_root))
    verb = "packed in shelf" if args.task == "shelf" else "stacked"

    n_ok = 0
    for e in envs:
        n_objects = sum(f.startswith("object_") for f in os.listdir(f"{env_root}/env_{e}"))
        rollout_dir = ROLLOUTS.format(task=TASK_DIR[args.task], split=args.split, env=e)
        final = run_env(task, env_root, e, rollout_dir) if os.path.exists(rollout_dir) else 0
        n_ok += final == n_objects
        print(f"env {e}: {final}/{n_objects} objects {verb}", flush=True)

    print(f"\n{args.task} {args.split}: success {n_ok}/{len(envs)} = {100 * n_ok / len(envs):.1f}%")


if __name__ == "__main__":
    main()
