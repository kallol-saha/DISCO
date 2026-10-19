"""
Propose, plan, execute and judge grasp-and-place candidates in the 32 parallel sims.
"""
import numpy as np
import torch

from disco.policy import models as M
from disco.tasks import PLANNING_FAILURES


def action_2x8(grasp7, place7):
    """Stack a grasp and a place pose into a (2, 8) action (gripper channel 1 = grasp, 0 = place).

    Args:
        grasp7: (7,) world-frame grasp pose (xyz + quaternion).
        place7: (7,) world-frame place pose (xyz + quaternion).
    """
    g = torch.cat([grasp7.float().cpu(), torch.ones(1)])
    p = torch.cat([place7.float().cpu(), torch.zeros(1)])
    return torch.stack([g, p])


def allocate(n_objects, K, B, rng):
    """Split the candidate budget over objects as places per grasp, with K grasps per object.

    With more objects than B // K, a random subset of B // K objects gets one place each.

    Args:
        n_objects: Number of movable objects.
        K: Grasps per object.
        B: Total number of candidates.
        rng: numpy random Generator.

    Returns:
        Dict {object position: places per grasp}, with places summing to B // K.
    """
    if K * n_objects > B:
        keep = sorted(rng.choice(n_objects, size=B // K, replace=False).tolist())
        return {i: 1 for i in keep}
    base = B // (K * n_objects)
    alloc = {i: base for i in range(n_objects)}
    extra = (B - base * K * n_objects) // K
    for i in rng.choice(n_objects, size=extra, replace=False).tolist():
        alloc[i] += 1
    return alloc


def propose(task, env, models, B=32, K=2, rng=None):
    """Sample B (grasp, place) candidates from the policy at the env's current state.

    Args:
        task: Task adapter from disco.tasks.
        env: Simulator with every slot at the parent state.
        models: Dict with "grasp" and "place" policy models.
        B: Total number of candidates.
        K: Grasps per object.
        rng: numpy random Generator (a fresh one if None).

    Returns:
        (candidates [{oi, grasp7, place7}], world-frame value cloud).
    """
    rng = rng if rng is not None else np.random.default_rng()
    objs = task.movable_objects(env)
    value_pcd = task.value_cloud(env)
    if not objs:
        return [], value_pcd
    cands = []
    for pos, places in allocate(len(objs), K, B, rng).items():
        oi = objs[pos]
        pcd = task.policy_cloud(env, oi)
        grasps = M.sample_grasps(models["grasp"], pcd, K).repeat_interleave(places, dim=0)
        place = M.sample_places(models["place"], pcd, grasps)
        cands += [{"oi": oi, "grasp7": grasps[j], "place7": place[j]} for j in range(len(grasps))]
    return cands, value_pcd


def score_candidates(models, value_pcd, cands):
    """Score candidates with the value network Q.

    Args:
        models: Dict with a "value" model.
        value_pcd: (P, 3) world-frame scene cloud.
        cands: Candidate dicts with grasp7 and place7.

    Returns:
        (len(cands),) float32 Q values in [0, 1].
    """
    if not cands:
        return np.zeros(0, dtype=np.float32)
    acts = torch.stack([action_2x8(c["grasp7"], c["place7"]) for c in cands])
    return M.score(models["value"], value_pcd, acts)


def _out(c, state, ftype):
    """Build the outcome dict of one candidate.

    Args:
        c: Candidate dict {oi, grasp7, place7}.
        state: Sim state after the attempt (the parent state if nothing ran).
        ftype: Failure type, or None on success.

    Returns:
        Dict with oi, action_2x8, state, failure_type, planning_failed, exec_failed.
    """
    return {"oi": c["oi"], "action_2x8": action_2x8(c["grasp7"], c["place7"]), "state": state,
            "failure_type": ftype, "planning_failed": ftype in PLANNING_FAILURES,
            "exec_failed": ftype is not None and ftype not in PLANNING_FAILURES}


def _pad(traj, T):
    """Pad trajectories to T steps by holding the last waypoint.

    Args:
        traj: (n, t, 8) joint + gripper trajectories.
        T: Target number of steps.
    """
    if traj.shape[1] == T:
        return traj
    return np.concatenate([traj, np.repeat(traj[:, -1:], T - traj.shape[1], axis=1)], axis=1)


def run_plans(env, items):
    """Execute and settle motion plans from their own start states, packed into the sim slots.

    Args:
        env: 32-slot simulator.
        items: List of (plan, start_states) pairs; plan["trajectories"] is (n, T, 8), start_states (n, D).

    Returns:
        End states (n, D) of each plan, in input order.
    """
    n_env, ends, chunks = env.num_envs, [None] * len(items), [[]]
    for k, (_, st) in enumerate(items):
        if sum(items[c][1].shape[0] for c in chunks[-1]) + st.shape[0] > n_env:
            chunks.append([])
        chunks[-1].append(k)
    for ch in [c for c in chunks if c]:
        states = torch.cat([items[k][1] for k in ch])
        T = max(items[k][0]["trajectories"].shape[1] for k in ch)
        batch = states[:1].repeat(n_env, 1).clone()
        batch[:len(states)] = states
        env.set_state(batch)
        env.robot_controller.execute_trajectory(np.concatenate([_pad(items[k][0]["trajectories"], T) for k in ch]))
        settle = max(items[k][0]["settle_steps"] for k in ch)
        if settle:
            env.wait_for_stability(max_steps=settle)
        end, o = env.get_state()[:len(states)].clone(), 0
        for k in ch:
            n = items[k][1].shape[0]
            ends[k], o = end[o:o + n], o + n
    return ends


def run_and_judge(env, items, judge):
    """Execute plans with `run_plans`, then judge each one from its end states.

    Args:
        env: 32-slot simulator.
        items: List of (plan, start_states) pairs.
        judge: Primitive judge; reads the plan's end states from the sim's first rows.

    Returns:
        One judge outcome per plan, in input order.
    """
    out = []
    for (plan, _), end in zip(items, run_plans(env, items)):
        batch = end[:1].repeat(env.num_envs, 1).clone()
        batch[:len(end)] = end
        env.set_state(batch)
        out.append(judge(plan))
    return out


def _rows(fails, params):
    """Map a primitive's failures back to their input rows by matching action params.

    Args:
        fails: List of (action params, failure_type, state, trajectory) from the primitive.
        params: (n, d) action params given to the primitive.

    Returns:
        Dict {row: (failure_type, state, trajectory)} for the failed rows; identical rows match in order.
    """
    rows = {}
    for f_act, f_type, f_state, f_traj in fails:
        for r in range(params.shape[0]):
            if r not in rows and torch.equal(f_act.cpu(), params[r].cpu()):
                rows[r] = (f_type, f_state, f_traj)
                break
    return rows


def execute(task, env, parent_state, cands, replicate=False):
    """Plan and execute the candidates of one tree node.

    Args:
        task: Task adapter (shelf or stacking) from disco.tasks.
        env: 32-slot simulator with motion primitives.
        parent_state: Flat sim state of the node.
        cands: Candidate dicts {oi, grasp7, place7}.
        replicate: Treat the candidates as repeated attempts of one action.

    Returns:
        One outcome dict per candidate, in input order.
    """
    return execute_many(task, env, [(parent_state, cands)], replicate)[0]


def execute_many(task, env, jobs, replicate=False):
    """Plan and execute the candidates of several tree nodes in the 32 sim slots.

    Args:
        task: Task adapter (shelf or stacking) from disco.tasks.
        env: 32-slot simulator with motion primitives.
        jobs: List of (parent_state, candidates) pairs, one per node.
        replicate: Treat each job's candidates as repeated attempts of one action.

    Returns:
        One list of outcome dicts per job, in input order.
    """
    n_env, ap = env.num_envs, env.action_primitives
    rep = lambda st, n=n_env: st.unsqueeze(0).repeat(n, 1)
    outs = [[None] * len(c) for _, c in jobs]

    # 1. per node (its own collision world): unique grasps, IK check, Pick plans
    gids, picks = [], []                       # picks: (job, unique-grasp ids, grasp params, plan)
    for j, (state, cands) in enumerate(jobs):
        keys, uniq, gid = {}, [], []
        for i, c in enumerate(cands):
            k = (i,) if replicate else (c["oi"], tuple(np.round(c["grasp7"].numpy(), 6)))
            if k not in keys:
                keys[k] = len(uniq)
                uniq.append(c)
            gid.append(keys[k])
        gids.append(gid)
        if not uniq:
            continue
        env.set_state(rep(state))
        grasps_robot = torch.stack([u["grasp7"] for u in uniq]).float().to("cuda:0")
        grasps_robot[:, 0] += M.ROBOT_BASE_X_OFFSET
        ap.update_collision_world()
        _, ik_ok = ap.inverse_kinematics(grasps_robot)
        ik_ok = ik_ok.reshape(len(uniq), -1)[:, 0].cpu().bool()
        for i, c in enumerate(cands):
            if not bool(ik_ok[gid[i]]):
                outs[j][i] = _out(c, state.clone(), "ik_infeasible")
        feas = [u for u in range(len(uniq)) if bool(ik_ok[u])]
        for s0 in range(0, len(feas), n_env):
            ids = feas[s0:s0 + n_env]
            gp = torch.zeros((len(ids), 9), device="cuda:0")
            for r, u in enumerate(ids):
                gp[r, :7] = uniq[u]["grasp7"].to("cuda:0")
                gp[r, 8] = float(uniq[u]["oi"])
            env.set_state(rep(state))
            try:
                plan = ap.plan_pick(gp)
            except RuntimeError as err:  # cuRobo plan_batch occasionally raises internal shape errors
                print(f"  [execute] pick planning raised {type(err).__name__}: {err}; treating as motion-plan failure")
                plan = {"result": ([], [], [], [], [(gp[r], "grasp_motion_plan", state, None) for r in range(len(ids))])}
            picks.append((j, ids, gp, plan))

    # 2. every node's picks execute together
    run = [(plan, rep(jobs[j][0], len(ids))) for j, ids, _, plan in picks if "result" not in plan]
    judged = iter(run_and_judge(env, run, ap.judge_pick))
    post, pick_traj, pick_fail = {}, {}, {}    # keyed by (job, unique-grasp id)
    for j, ids, gp, plan in picks:
        ok_states, ok_trajs, _, _, fails = plan["result"] if "result" in plan else next(judged)
        failed = _rows(fails, gp)
        k = 0
        for r, u in enumerate(ids):
            if r in failed:
                ftype, st, _ = failed[r]
                pick_fail[(j, u)] = (ftype, st.clone())
            else:
                post[(j, u)], pick_traj[(j, u)] = ok_states[k], ok_trajs[k]
                k += 1

    # 3. Place plans per held grasp, from that grasp's post-pick state. Place / PlaceStacking attach the
    #    held object to the planner's robot model from env slot 0, so every row of a plan must hold the
    #    object the same way (same grasp, same post-grasp state).
    places = []                                # (job, candidate indices, place params, plan, start states)
    for j, (state, cands) in enumerate(jobs):
        gid = gids[j]
        for i, c in enumerate(cands):
            if outs[j][i] is None and (j, gid[i]) in pick_fail:
                ftype, st = pick_fail[(j, gid[i])]
                outs[j][i] = _out(c, st, ftype)
        todo = [i for i in range(len(cands)) if outs[j][i] is None]
        group = [0] * len(cands) if replicate else gid
        for g in sorted({group[i] for i in todo}):
            idx = [i for i in todo if group[i] == g]
            oi = cands[idx[0]]["oi"]
            for s0 in range(0, len(idx), n_env):
                chunk = idx[s0:s0 + n_env]
                states = torch.stack([post[(j, gid[i])] for i in chunk])
                batch = rep(state.to(states.device)).clone()
                batch[:len(chunk)] = states
                env.set_state(batch)
                pp = torch.zeros((len(chunk), 8), device="cuda:0")
                for r, i in enumerate(chunk):
                    pp[r, :7] = cands[i]["place7"].to("cuda:0")
                    pp[r, 7] = float(oi)
                if getattr(task, "correct_inhand", False) and "place_obj7" in cands[chunk[0]]:
                    # The object shifts in the hand during the pick; re-target the gripper so the object
                    # (not the nominal grasp) lands on the intended pose: gripper = obj_place ∘ (obj -> EE),
                    # with obj -> EE measured in the post-grasp state (row 0 holds it; rows share the grasp).
                    from disco.motion.action_primitives import EE_LINK_CENTER_TO_GRIPPER_TIP
                    from disco.utils.pose_utils import compose_pose, inv_pose, quat_rotate
                    tcp = env.get_gripper_pose()[0].cpu().numpy().astype(np.float64)
                    ee = tcp.copy()
                    ee[:3] = tcp[:3] + quat_rotate(tcp[3:], np.array([0.0, 0.0, -EE_LINK_CENTER_TO_GRIPPER_TIP]))
                    obj = env.object_poses_tensor[0, oi].cpu().numpy().astype(np.float64)
                    o2e = compose_pose(inv_pose(obj), ee)
                    for r, i in enumerate(chunk):
                        pp[r, :7] = torch.tensor(compose_pose(cands[i]["place_obj7"], o2e), dtype=torch.float32,
                                                 device="cuda:0")
                try:
                    plan = task.plan_place(env, pp)
                except RuntimeError as err:  # cuRobo plan_batch internal shape errors
                    print(f"  [execute] place planning raised {type(err).__name__}: {err}; treating as motion-plan failure")
                    plan = {"result": ([], [], [], [], [(pp[r], "place_motion_plan", state, None)
                                                        for r in range(len(chunk))])}
                places.append((j, chunk, pp, plan, states))

    # 4. every node's places execute together
    run = [(plan, states) for _, _, _, plan, states in places if "result" not in plan]
    judged = iter(run_and_judge(env, run, lambda plan: task.judge_place(env, plan)))
    for j, chunk, pp, plan, _ in places:
        state, gid = jobs[j][0], gids[j]
        ok_states, ok_trajs, _, _, fails = plan["result"] if "result" in plan else next(judged)
        failed = _rows(fails, pp)
        k = 0
        for r, i in enumerate(chunk):
            c = jobs[j][1][i]
            if r in failed:
                ft, st, tr = failed[r]
                outs[j][i] = _out(c, state.clone() if ft in PLANNING_FAILURES else st.clone(), ft)
                if tr is not None:  # executed but judged failed: keep it so callers can re-judge
                    outs[j][i]["trajectory"] = np.concatenate([pick_traj[(j, gid[i])], tr], axis=0)
            else:
                outs[j][i] = _out(c, ok_states[k].clone(), None)
                # Joint + gripper trajectory of the whole action (pick, then place), for replay.
                outs[j][i]["trajectory"] = np.concatenate([pick_traj[(j, gid[i])], ok_trajs[k]], axis=0)
                k += 1

    if jobs:
        env.set_state(rep(jobs[0][0]))
    return outs
