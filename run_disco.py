"""
Run DISCO: rounds of MCTS search and distillation of the policy and value networks.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
OK = (0, -11, 139)  # SIGSEGV (-11 from subprocess, 139 from a shell): SAPIEN at exit, after outputs are written


def run_pool(jobs, gpus, per_gpu, log_dir, retries=3):
    """Run subprocess jobs over the GPUs, at most `per_gpu` at a time on each, retrying failures.

    Args:
        jobs: List of (name, argv) pairs; each job logs to `<log_dir>/<name>.log`.
        gpus: GPU indices to spread the jobs over.
        per_gpu: Max concurrent jobs per GPU.
        log_dir: Directory for the per-job logs.
        retries: Attempts per job before it counts as failed.

    Returns:
        Names of the jobs that failed every attempt.
    """
    slots = [g for g in gpus for _ in range(per_gpu)]
    queue, lock, failed = list(jobs), threading.Lock(), []

    def worker(gpu):
        """Take jobs from the shared queue and run them on one GPU until the queue is empty.

        Args:
            gpu: GPU index, exported as CUDA_VISIBLE_DEVICES.
        """
        while True:
            with lock:
                if not queue:
                    return
                name, argv = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT))
            for _ in range(retries):
                with open(log_dir / f"{name}.log", "a") as log:
                    rc = subprocess.call(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if rc in OK:
                    break
            else:
                failed.append(name)

    threads = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def train(kind, data, init, models, run, iters, gpu, log, channels, use_wandb=False):
    """Train one network in a subprocess, starting from `init` if given.

    Args:
        kind: "grasp" or "place" (policy) or "value".
        data: Training dataset (.pth).
        init: Checkpoint to start from, or None to train from scratch.
        models: Output directory; the run is saved under `models/<run>`.
        run: Run name (also the wandb run name).
        iters: Training iterations.
        gpu: GPU index to train on.
        log: Log file the trainer output is appended to.
        channels: Policy point-cloud input channels (3 for shelf, 4 for stacking).
        use_wandb: Log training to wandb.

    Returns:
        Path to the run's last.pth checkpoint.
    """
    val = next(f for f in (1000, 500, 100, 50, 10, 1) if iters % f == 0)
    common = ["--train_data_dir", str(data), "--base_log_dir", str(models.parent),
              "--wandb", str(use_wandb), "--wandb_project_name", models.name, "--wandb_run_name", run,
              "--checkpoint", str(Path(init).resolve()) if init else "null", "--train_iters", str(iters),
              # last.pth is written at validation steps
              "--val_freq", str(val), "--vis_freq", str(iters), "--interm_ckpt_freq", str(iters)]
    if kind == "value":
        argv = [sys.executable, "-m", "disco.flowmatch.train_value", *common]
    else:
        argv = [sys.executable, "-m", "disco.flowmatch.train_policy", "--mode", kind,
                "--pcd_input_channels", str(channels), *common]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT))
    with open(log, "a") as f:
        rc = subprocess.call(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    ckpt = models / run / "last.pth"
    if rc not in OK or not ckpt.exists():
        raise RuntimeError(f"training {kind} failed (rc={rc}); see {log}")
    return ckpt


def budget_used(path):
    """Return the simulator calls used so far over all envs (0 if the budget file does not exist yet).

    Args:
        path: Path to the shared sim_budget.json.
    """
    return sum(json.load(open(path))["used"].values()) if path.exists() else 0


def main():
    """Run DISCO rounds (search, extract data, distill) until the simulation budget is spent."""
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "config" / "run_disco.yaml"), OmegaConf.from_cli())
    from disco.tasks import get_task
    task = get_task(cfg.task)
    env_root, budget, gpus = cfg.env_root, cfg.budget, list(cfg.gpus)
    per_gpu = min(cfg.per_gpu, 4)
    envs = (list(cfg.envs) if cfg.envs is not None else
            sorted(int(d.name.split("_")[1]) for d in Path(env_root).glob("env_*")))
    out = Path(cfg.out).resolve()
    logs, budget_file = out / "logs", out / "sim_budget.json"   # β is shared by all envs and rounds
    logs.mkdir(parents=True, exist_ok=True)

    state_path = out / "state.json"
    state = json.load(open(state_path)) if state_path.exists() else {
        "round": 1, "grasp": str(Path(cfg.grasp).resolve()), "place": str(Path(cfg.place).resolve()),
        "value": str(Path(cfg.value).resolve()) if cfg.value else None, "used_at_start": 0}

    while True:
        r = state["round"]
        trees = out / "trees" / f"round_{r}"   # every round searches fresh trees from the root
        print(f"=== round {r}: search {cfg.block} iters/env "
              f"(budget used {budget_used(budget_file)}/{budget})", flush=True)
        search = [(f"r{r}_search_env{e}", [
            sys.executable, "-m", "disco.search.run", f"task={cfg.task}", f"env_index={e}",
            f"env_path={env_root}", f"iterations={cfg.block}", f"paths.trees_base={trees}",
            f"sim_budget={budget}", f"sim_budget_path={budget_file}", f"grasp_only_checkpoint={state['grasp']}",
            f"place_conditional_checkpoint={state['place']}",
            f"value_checkpoint={state['value'] or 'null'}",
            f"mcts.parallel_descents={cfg.parallel_descents}"]) for e in envs]
        failed = run_pool(search, gpus, per_gpu, logs)
        if failed:
            print(f"  search failed for {failed} (kept going)", flush=True)

        data = out / "data" / f"round_{r}"
        extract = [(f"r{r}_extract_env{e}", [
            sys.executable, "-m", "disco.data", "--task", cfg.task, "--env_root", env_root,
            "--trees", str(trees), "--env_index", str(e), "--out", str(data)])
            for e in envs if (trees / f"env_{e}" / "node_0.npz").exists()]
        run_pool(extract, gpus, per_gpu, logs)
        subprocess.check_call([sys.executable, "-m", "disco.data", "--out", str(data), "--merge"], cwd=ROOT,
                              env=dict(os.environ, PYTHONPATH=str(ROOT)))

        # unique run names keep wandb from resuming an old run
        models = out / "models" / "DISCO_distill"
        log = logs / f"r{r}_train.log"
        results = {}
        jobs = [("grasp", data / "policy.pth", state["grasp"], cfg.policy_iters, gpus[0]),
                ("place", data / "policy.pth", state["place"], cfg.policy_iters, gpus[-1]),
                ("value", data / "value.pth", state["value"], cfg.value_iters, gpus[0])]
        jobs = [j for j in jobs if j[1].exists()]   # a model with no data this round keeps its checkpoint
        if not jobs and budget_used(budget_file) == state.get("used_at_start", 0):
            raise RuntimeError(f"round {r}: no simulation and no training data; check the policies ({logs})")

        def fit(kind, d, init, iters, gpu):
            """Train one model for this round and store its checkpoint in `results`.

            Args:
                kind: "grasp", "place" or "value".
                d: Training dataset (.pth).
                init: Checkpoint to start from, or None.
                iters: Training iterations.
                gpu: GPU index to train on.
            """
            results[kind] = train(kind, d, init, models, f"{out.name}_r{r}_{kind}", iters, gpu, log,
                                  task.policy_channels, cfg.wandb)

        threads = [threading.Thread(target=fit, args=j) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if len(results) < len(jobs):
            raise RuntimeError(f"round {r} distillation incomplete; see {log}")

        state.update(round=r + 1, used_at_start=budget_used(budget_file),
                     **{k: str(v) for k, v in results.items()})
        json.dump(state, open(state_path, "w"), indent=1)
        print(f"  round {r} distilled: {sorted(results) or 'nothing (no data)'}", flush=True)
        if budget_used(budget_file) >= budget:
            final = {k: state[k] for k in ("grasp", "place", "value")}
            print(f"=== simulation budget spent ({budget_used(budget_file)}/{budget}); final models: {final}")
            return


if __name__ == "__main__":
    main()
