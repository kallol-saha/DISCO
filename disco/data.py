"""
Turn MCTS trees into policy and value training datasets.
"""
import argparse
import glob
import os

import numpy as np
import torch

from disco.tasks import get_task
from disco.search.mcts import MCTS

GOAL = 1.0 - 1e-6


def leaf_stats(root, is_success):
    """Count the successful and total visited leaves below every node.

    Args:
        root: Root node of the loaded tree.
        is_success: Predicate on a leaf node.

    Returns:
        Dict {node id: (n_success, n_leaves)}.
    """
    stats = {}

    def visit(node):
        """Fill `stats` for the subtree of `node` and return its (n_success, n_leaves).

        Args:
            node: Tree node to visit.
        """
        kids = [c for c in node.children if c.visit_count > 0 and not c.is_failed]
        if not kids:
            stats[node.id] = (int(is_success(node)), 1)
        else:
            s = [visit(c) for c in kids]
            stats[node.id] = (sum(a for a, _ in s), sum(b for _, b in s))
        return stats[node.id]

    visit(root)
    return stats


def action_2x8(prev_action):
    """Convert a node action to the (2, 8) training action (grasp row g=1, place row g=0).

    Args:
        prev_action: (17,) node action [grasp7, cost, obj, place7, gripper].
    """
    a = prev_action.float().cpu()
    return torch.stack([torch.cat([a[0:7], torch.ones(1)]), torch.cat([a[9:16], torch.zeros(1)])])


def process_env(task, env_root, trees, env_index, success):
    """Build the policy and value rows from one env's search tree.

    Args:
        task: Task adapter from disco.tasks.
        env_root: Directory of env scenes (env_<i>/).
        trees: Directory holding the env_<i>/ trees.
        env_index: Env to process.
        success: "goal" (leaf reaches the goal) or "best" (leaf matches the env's best return).

    Returns:
        (policy, value) dicts of stacked tensors; each is None if it has no rows.
    """
    env = task.make_env(env_root, env_index)
    planner = MCTS(simulator=env, nodes_dir=os.path.join(trees, f"env_{env_index}"))
    root = planner.loaded_root_node
    ret = {n.id: MCTS.path_return(n) for n in planner.nodes}
    best = max(ret[n.id] for n in planner.nodes if not n.is_failed)
    thresh = GOAL if success == "goal" else best - 1e-6
    stats = leaf_stats(root, lambda n: (not n.is_failed) and ret[n.id] >= thresh and ret[n.id] > 0)

    pol, val = [], []
    for parent in planner.nodes:
        kids = [c for c in parent.children if not c.is_failed and c.prev_action is not None]
        if not kids:
            continue
        env.set_state(parent.state.unsqueeze(0).repeat(env.num_envs, 1))
        value_pcd = torch.from_numpy(task.value_cloud(env))
        policy_pcd = {}
        for c in kids:
            a = action_2x8(c.prev_action)
            oi = int(c.prev_action[8].item())
            if oi not in policy_pcd:
                policy_pcd[oi] = torch.from_numpy(task.policy_cloud(env, oi))
            pol.append((policy_pcd[oi], a, c.id))
            if c.visit_count > 0:
                n_succ, n_leaf = stats[c.id]
                val.append((value_pcd, a, n_succ / n_leaf, n_leaf, c.visit_count, parent.id, c.id))
    env.close()
    print(f"env {env_index}: {len(planner.nodes)} nodes, best return {best:.3f}, "
          f"{len(pol)} policy rows, {len(val)} value rows, root y = {stats[0][0]}/{stats[0][1]}")

    def col(rows, i, dtype=None):
        """Stack field `i` of every row into one tensor.

        Args:
            rows: List of row tuples.
            i: Field index.
            dtype: Tensor dtype, used when the field holds scalars.
        """
        x = [r[i] for r in rows]
        return torch.stack(x) if torch.is_tensor(x[0]) else torch.tensor(x, dtype=dtype)

    e = lambda rows: torch.full((len(rows),), env_index, dtype=torch.int32)
    policy = {"input_pcd": col(pol, 0), "goal_pose": col(pol, 1), "env_index": e(pol),
              "node_id": col(pol, 2, torch.int32)} if pol else None
    value = {"input_pcd": col(val, 0), "goal_pose": col(val, 1), "target": col(val, 2, torch.float32),
             "n_leaves": col(val, 3, torch.int32), "visit_count": col(val, 4, torch.int32),
             "env_index": e(val), "parent_id": col(val, 5, torch.int32),
             "node_id": col(val, 6, torch.int32)} if val else None
    return policy, value


def merge(out):
    """Concatenate the per-env shards into policy.pth and value.pth.

    Args:
        out: Data directory holding shards/.
    """
    for kind in ("policy", "value"):
        shards = sorted(glob.glob(os.path.join(out, "shards", f"{kind}_env_*.pth")))
        blobs = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
        if not blobs:
            continue
        merged = {k: torch.cat([b[k] for b in blobs]) for k in blobs[0]}
        torch.save(merged, os.path.join(out, f"{kind}.pth"))
        print(f"{out}/{kind}.pth: {merged['input_pcd'].shape[0]} rows from {len(blobs)} envs")


def main():
    """Write one env's dataset shards, or merge all shards with --merge."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=["shelf", "stacking"])
    ap.add_argument("--env_root")
    ap.add_argument("--trees")
    ap.add_argument("--env_index", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--success", choices=["goal", "best"], default="goal")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge(args.out)

    policy, value = process_env(get_task(args.task), args.env_root, args.trees, args.env_index, args.success)
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)
    for kind, blob in (("policy", policy), ("value", value)):
        if blob is not None:
            torch.save(blob, os.path.join(args.out, "shards", f"{kind}_env_{args.env_index}.pth"))


if __name__ == "__main__":
    main()
