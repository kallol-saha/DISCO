"""
Expand MCTS nodes: propose candidates with the policy, execute them, score with the task reward.
"""
from disco.search import candidates as C
from disco.policy import models as M
from disco.policy.shelf_sampler import ActionSampler


def propose(task, env, state, K, P, B):
    """Sample (grasp, place) candidates from the policy at one tree state.

    Args:
        task: Task adapter from disco.tasks.
        env: 32-slot simulator with an action sampler.
        state: Flat sim state of the node.
        K: Grasps per object.
        P: Places per grasp (shelf only).
        B: Total number of candidates.

    Returns:
        (candidates [{oi, grasp7, place7}], full-scene xyz cloud in the robot frame for Q).
    """
    if task.name == "shelf":
        flat, value_pcd_robot = env.action_sampler.sample_grasp_place_actions(
            state, num_grasps_per_object=K, num_places_per_grasp=P, total_budget=B,
            return_candidates_only=True)
        return [{"oi": c["oi"], "grasp7": c["grasp7"], "place7": p}
                for c in flat for p in c["place7_list"]], value_pcd_robot
    env.set_state(state.unsqueeze(0).repeat(env.num_envs, 1))
    cands, value_pcd = C.propose(task, env, env.action_sampler.models, B=B or 32, K=K)
    value_pcd_robot = value_pcd.copy()
    value_pcd_robot[:, 0] += M.ROBOT_BASE_X_OFFSET
    return cands, value_pcd_robot


def expand(task, env, states, K=2, P=2, B=32):
    """Propose, plan and execute candidates at several tree states in one batch.

    Args:
        task: Task adapter from disco.tasks.
        env: 32-slot simulator with an action sampler.
        states: Flat sim states of the nodes to expand.
        K: Grasps per object.
        P: Places per grasp (shelf only).
        B: Candidates per state.

    Returns:
        One (results, value_pcd_robot) pair per state, in input order. Each result dict holds
        state, action (17,), action_2x8, trajectory, reward (task-reward gain), is_failed, failure_type.
    """
    proposals = [propose(task, env, s, K, P, B) for s in states]
    outcomes = C.execute_many(task, env, [(s, cands) for s, (cands, _) in zip(states, proposals)])
    expansions = []
    for state, (cands, value_pcd_robot), outs in zip(states, proposals, outcomes):
        r0 = _reward(task, env, state)
        results = []
        for c, o in zip(cands, outs):
            ok = o["failure_type"] is None
            results.append({
                "state": o["state"],
                "action": ActionSampler._make_action_17(c["grasp7"], c["place7"], c["oi"]),
                "action_2x8": o["action_2x8"],
                "trajectory": o.get("trajectory"),
                "reward": _reward(task, env, o["state"]) - r0 if ok else 0.0,
                "is_failed": not ok,
                "failure_type": o["failure_type"],
            })
        expansions.append((results, value_pcd_robot))
    return expansions


def _reward(task, env, state):
    """Return the task reward of a state, after setting it in every sim slot.

    Args:
        task: Task adapter from disco.tasks.
        env: 32-slot simulator.
        state: Flat sim state.
    """
    env.set_state(state.unsqueeze(0).repeat(env.num_envs, 1))
    return task.reward(env)
