"""
Run MCTS on one environment.
"""
import os
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from disco.tasks import get_task
from disco.search.budget import SimBudget
from disco.search.mcts import MCTS, Node


def build_env(cfg):
    """Build the 32-slot sim and load the grasp and place policies into its action sampler.

    Args:
        cfg: Hydra config (task, env, checkpoints, devices).
    """
    env = get_task(cfg.task).make_env(cfg.env_path, cfg.env_index, num_envs=cfg.num_envs)
    if cfg.task == "stacking":
        from disco.policy.stacking_sampler import StackingPolicySampler
        env.action_sampler = StackingPolicySampler(env)
    env.action_sampler.load_policy("grasp_only", cfg.grasp_only_checkpoint, cfg.policy_device)
    env.action_sampler.load_policy("place_conditional", cfg.place_conditional_checkpoint, cfg.policy_device)
    return env


@hydra.main(version_base=None, config_path="../../config", config_name="mcts")
def main(cfg: DictConfig):
    """Run (or resume) MCTS on one env and save its tree.

    Args:
        cfg: Hydra config from config/mcts.yaml plus command-line overrides.
    """
    nodes_dir = os.path.join(cfg.paths.trees_base, f"env_{cfg.env_index}")
    os.makedirs(nodes_dir, exist_ok=True)
    print(f"[{cfg.task}] env {cfg.env_index} -> {nodes_dir}")

    env = build_env(cfg)
    planner = MCTS(
        simulator=env,
        **OmegaConf.to_container(cfg.mcts, resolve=True),
        max_depth=env.num_objects,          # task horizon K = number of objects
        nodes_dir=nodes_dir,
        q_function_checkpoint=cfg.value_checkpoint,
        q_function_device=cfg.value_device,
        task=get_task(cfg.task),
    )
    root = Node(id=0, simulator=env, state=env.get_state()[0], C=cfg.mcts.C)

    budget = None
    if cfg.sim_budget is not None or cfg.sim_budget_path is not None:
        budget = SimBudget(cfg.sim_budget_path or os.path.join(cfg.paths.trees_base, "sim_budget.json"),
                           total=cfg.sim_budget)

    planner.plan(root, iterations=cfg.iterations, budget=budget, env_index=cfg.env_index)
    print(f"[{cfg.task}] env {cfg.env_index} done: {len(planner.nodes)} nodes in {nodes_dir}")
    env.close()


if __name__ == "__main__":
    main()
