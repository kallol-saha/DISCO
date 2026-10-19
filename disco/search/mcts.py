"""
DISCO MCTS: progressive widening, UCB descent, Q-initialised children, return backup.
"""
import json
import os

import cv2
import numpy as np
import torch

from disco.envs.shelf_packing_base import ShelfPackingBase as Simulator

# Candidate failure types rejected before anything runs in the simulator.
NOT_SIMULATED = {"ik_infeasible", "pre_grasp_motion_plan", "grasp_motion_plan"}

class Node:
    """One search-tree vertex: a saved sim state plus its MCTS statistics."""

    def __init__(
        self,
        id: int,
        simulator: Simulator,
        state: torch.Tensor,
        parent=None,
        prev_action: torch.Tensor = None,
        prev_trajectory: torch.Tensor = None,
        reward: float = 0.0,
        C: float = 2.0,
        is_failed: bool = False,
        failure_type: str = None,
        ):
        """Create a tree vertex.

        Args:
            id: Unique node id (also names its file, node_<id>.npz).
            simulator: Simulator the state belongs to.
            state: Flat sim state of one env slot.
            parent: Parent node, or None for the root.
            prev_action: (17,) action that led here from the parent.
            prev_trajectory: Motion-planned trajectory of that action.
            reward: Task-reward gain on the edge into this node.
            C: UCB exploration constant.
            is_failed: Whether the action into this node failed.
            failure_type: Failure label, or None.
        """

        # --------------------------------------------- #

        self.id = id
        self.state = state
        self.parent = parent
        self.parent_id = parent.id if parent is not None else None
        self.simulator = simulator
        self.batch_size = self.simulator.num_envs
        self.prev_action = prev_action
        self.prev_trajectory = prev_trajectory
        self.reward = reward
        self.Q = reward
        self.Q_history = []
        self.C = C
        self.rollout_visits = 0
        self.rollout_value = 0.0
        self.is_failed = is_failed
        self.failure_type = failure_type
        self.created_at_iteration = -1  # Set by MCTS when creating the node
        # Iteration in which this node was first selected during a downward
        # walk (i.e. first time visit_count was bumped from 0). -1 if never
        # visited. Together with `created_at_iteration` this lets you
        # reconstruct, per iteration, which nodes were added vs. which were
        # actually entered by the search.
        self.first_visit_at_iteration = -1

        # Post-hoc evaluation values written by evaluate_value_on_trees.py.
        # predicted_value: V̂(parent_state, prev_action) from the value network.
        # subtree_value:   fraction of visited leaves below this node that end in success.
        self.predicted_value = None
        self.subtree_value = None

        # List of object indices (positions in simulator.object_names) sitting
        # inside the shelf at this node's saved state. None until populated.
        self.objects_in_shelf = None

        # Crash-recovery cursor. Exactly one node in a tree should hold this
        # flag at a time; it marks the descent frontier of an in-flight
        # iteration. Cleared at iter end. See `MCTS._set_frontier` /
        # `MCTS._clear_frontier`. Persisted in `Node.save` / `Node.load`.
        self.is_expansion_frontier = False

        # --------------------------------------------- #

        self.children: list = []
        self.children_ids = []
        self.visit_count: int = 0

        self.dirty = True  # New nodes need saving

    @property
    def depth(self) -> int:
        """Return the number of edges from the root to this node."""
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    @property
    def UCB(self):
        """Return the UCB score Q + C·sqrt(ln(N_parent+1) / (N+1)), or -inf if failed."""

        if self.is_failed:
            return float('-inf')

        n_parent = self.parent.visit_count if self.parent is not None else 0
        n_edge = self.visit_count

        ucb = self.Q + self.C * np.sqrt( np.log(n_parent+1) / (n_edge+1) )

        return ucb

    def capture_image(self):
        """Render this node's state and return it as an RGB image."""
        
        self.simulator.set_state(self.state.unsqueeze(0).repeat(self.simulator.num_envs, 1))
        if self.simulator.render_mode is not None:
            self.simulator.render()
        image = self.simulator.capture_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def save(self, nodes_dir: str):
        """Write the node to `<nodes_dir>/node_<id>.npz`.

        Args:
            nodes_dir: Directory to write into.
        """
        file_path = os.path.join(nodes_dir, f"node_{self.id}.npz")

        # Update children_ids from actual children list to ensure consistency
        self.children_ids = [child.id for child in self.children]

        # Update parent_id from parent if it exists
        if self.parent is not None:
            self.parent_id = self.parent.id

        # Prepare data dictionary
        data = {
            'id': self.id,
            'state': self.state.cpu().detach().numpy() if isinstance(self.state, torch.Tensor) else self.state,
            'parent_id': self.parent_id if self.parent_id is not None else -1,
            'reward': self.reward,
            'Q': self.Q,
            'C': self.C,
            'visit_count': self.visit_count,
            'children_ids': np.array(self.children_ids) if len(self.children_ids) > 0 else np.array([], dtype=np.int64),
            'rollout_value': self.rollout_value,
            'rollout_visits': self.rollout_visits,
            'Q_history': np.array(self.Q_history) if len(self.Q_history) > 0 else np.array([], dtype=np.float64),
            'is_failed': self.is_failed,
            'failure_type': self.failure_type if self.failure_type is not None else '',
            'created_at_iteration': self.created_at_iteration,
            'first_visit_at_iteration': self.first_visit_at_iteration,
        }

        if self.predicted_value is not None:
            data['predicted_value'] = float(self.predicted_value)
        if self.subtree_value is not None:
            data['subtree_value'] = float(self.subtree_value)

        if self.objects_in_shelf is not None:
            data['objects_in_shelf'] = np.array(self.objects_in_shelf, dtype=np.int64)

        data['is_expansion_frontier'] = bool(self.is_expansion_frontier)

        # Save prev_action if it exists
        if self.prev_action is not None:
            data['prev_action'] = self.prev_action.cpu().numpy() if isinstance(self.prev_action, torch.Tensor) else self.prev_action

        # Save prev_trajectory if it exists
        if self.prev_trajectory is not None:
            data['prev_trajectory'] = self.prev_trajectory.cpu().numpy() if isinstance(self.prev_trajectory, torch.Tensor) else self.prev_trajectory

        np.savez_compressed(file_path, **data)
    
    @classmethod
    def load(cls, file_path: str, simulator: Simulator, parent=None):
        """Read a node from an npz file written by `save`.

        Args:
            file_path: Path to the node_<id>.npz file.
            simulator: Simulator to attach to the node.
            parent: Parent node, if known (links are usually restored later by `load_nodes`).
        """
        data = np.load(file_path, allow_pickle=True)
        
        # Extract data
        node_id = int(data['id'])
        state = torch.tensor(data['state'], device='cuda:0')
        parent_id = int(data['parent_id']) if data['parent_id'] != -1 else None
        reward = float(data['reward'])
        Q = float(data['Q'])
        C = float(data['C'])
        visit_count = int(data['visit_count'])
        children_ids = data['children_ids'].tolist() if len(data['children_ids']) > 0 else []

        # Load optional fields
        prev_action = None
        if 'prev_action' in data:
            prev_action = torch.tensor(data['prev_action'], device='cuda:0')

        prev_trajectory = None
        if 'prev_trajectory' in data:
            prev_trajectory = data['prev_trajectory']

        # Create node
        node = cls(
            id=node_id,
            simulator=simulator,
            state=state,
            parent=parent,
            prev_action=prev_action,
            prev_trajectory=prev_trajectory,
            reward=reward,
            C=C
        )

        # Set additional attributes
        node.Q = Q
        node.visit_count = visit_count
        node.children_ids = children_ids

        # Load rollout_value and rollout_visits if they exist (for backward compatibility)
        if 'rollout_value' in data:
            node.rollout_value = float(data['rollout_value'])
        else:
            node.rollout_value = 0.0

        if 'rollout_visits' in data:
            node.rollout_visits = int(data['rollout_visits'])
        else:
            node.rollout_visits = 0
        node.parent_id = parent_id

        if 'Q_history' in data and len(data['Q_history']) > 0:
            node.Q_history = data['Q_history'].tolist()
        else:
            node.Q_history = []

        if 'is_failed' in data:
            node.is_failed = bool(data['is_failed'])
        else:
            node.is_failed = False

        if 'failure_type' in data:
            ft = str(data['failure_type'])
            node.failure_type = ft if ft != '' else None
        else:
            node.failure_type = None

        if 'created_at_iteration' in data:
            node.created_at_iteration = int(data['created_at_iteration'])
        else:
            node.created_at_iteration = -1

        # Default to -1 for old npz files that pre-date this field.
        if 'first_visit_at_iteration' in data:
            node.first_visit_at_iteration = int(data['first_visit_at_iteration'])
        else:
            node.first_visit_at_iteration = -1

        if 'predicted_value' in data:
            node.predicted_value = float(data['predicted_value'])
        if 'subtree_value' in data:
            node.subtree_value = float(data['subtree_value'])

        if 'objects_in_shelf' in data:
            node.objects_in_shelf = data['objects_in_shelf'].tolist()
        else:
            node.objects_in_shelf = None

        node.is_expansion_frontier = bool(data['is_expansion_frontier']) if 'is_expansion_frontier' in data else False

        node.dirty = False  # Loaded from disk, no changes yet

        return node


class MCTS:
    """MCTS with progressive widening over grasp-and-place actions, saved node by node to disk."""

    def __init__(
        self,
        simulator: Simulator,
        C: float = 2.0,
        alpha: float = 0.95,
        k: float = 1.0,
        max_depth: int = 10,
        nodes_dir: str = "assets/nodes",
        # Learned Q-function. None = no Q̂: fresh children seed Q with their
        # progress reward (first distillation round).
        q_function_checkpoint: str = None,
        q_function_device: str = "cuda:0",
        # Expansion branching for the cascaded grasp+place policy: K =
        # `num_grasps_per_object` grasps per non-shelved object, each with
        # `num_places_per_grasp` placements. With `total_budget=B`, K stays
        # fixed and placements per grasp are derived so one expansion emits
        # ~B candidates regardless of object count (see
        # ActionSampler.sample_grasp_place_actions).
        num_grasps_per_object: int = 2,
        num_places_per_grasp: int = 2,
        total_budget: int = None,
        # "ucb" picks the child with the highest UCB (default). "random" picks
        # uniformly among children — with q_function_checkpoint=None this is
        # the pure-policy baseline (no Q̂, no UCB ranking).
        selection_mode: str = "ucb",
        # Task adapter (disco.tasks.ShelfTask / StackingTask) used for expansion.
        task=None,
        # Descents run together per iteration batch. Their frontier vertices
        # are expanded together: cuRobo plans one vertex at a time, the
        # simulator executes all of their plans in one batch. 1 = sequential.
        parallel_descents: int = 1,
        ):
        """Set up the search and load any tree already saved in `nodes_dir`.

        Args:
            simulator: 32-slot simulator with motion primitives.
            C: UCB exploration constant.
            alpha: Progressive-widening exponent.
            k: Progressive-widening coefficient (expand while |children| < k·N^alpha).
            max_depth: Maximum descent depth.
            nodes_dir: Directory holding the node_<id>.npz files.
            q_function_checkpoint: Value-network checkpoint for Q̂, or None to seed Q with the reward.
            q_function_device: Device for the value network.
            num_grasps_per_object: Grasps sampled per movable object in one expansion.
            num_places_per_grasp: Placements sampled per grasp.
            total_budget: Target candidates per expansion, or None for the task default.
            selection_mode: "ucb" or "random" child selection.
            task: Task adapter (shelf or stacking) from disco.tasks.
            parallel_descents: Descents run together per batch (1 = sequential).
        """

        self.simulator = simulator
        self.nodes: list[Node] = []
        self.nodes_dir = nodes_dir
        self.loaded_root_node = None

        self.C = C
        self.alpha = alpha
        self.k = k
        self.max_depth = max_depth
        self.num_grasps_per_object = num_grasps_per_object
        self.num_places_per_grasp = num_places_per_grasp
        self.total_budget = total_budget
        if selection_mode not in ("ucb", "random"):
            raise ValueError(
                f"selection_mode must be 'ucb' or 'random', got {selection_mode!r}"
            )
        self.selection_mode = selection_mode
        self.task = task
        self.parallel_descents = max(1, int(parallel_descents))
        self.current_iteration = -1
        self.sim_calls_last_iter = 0

        self.q_function = None
        self.q_function_device = torch.device(q_function_device)
        if q_function_checkpoint is not None:
            self.q_function = self._load_q_function(q_function_checkpoint)
        else:
            print("Q-function disabled (no checkpoint provided); "
                  "fresh children seed Q with reward only")

        # Per-iteration log: maps iteration -> {leaf_node_id, leaf_predicted_value}
        self.iteration_log: dict = {}

        # Crash-recovery cursor. Set after `load_nodes` if a previously
        # interrupted iteration left a frontier flag on disk. Consumed by
        # `mcts_iteration` on the next iter to skip the initial
        # `_increment_visit(root)` and resume descent at the flagged node.
        self._resume_node_id: int = None
        self._resume_iteration: int = None
        self._in_flight_path = None  # set after nodes_dir exists

        # Create nodes directory if it doesn't exist
        os.makedirs(self.nodes_dir, exist_ok=True)

        self._in_flight_path = os.path.join(self.nodes_dir, "in_flight.json")

        # Load existing nodes if directory has node files
        self.loaded_root_node = self.load_nodes()
        # Detect a previously-interrupted iteration via the saved frontier
        # flag + the in-flight sidecar file.
        self._detect_resume_state()

        # Load iteration log if it exists
        self._iteration_log_path = os.path.join(self.nodes_dir, "iteration_log.json")
        if os.path.exists(self._iteration_log_path):
            with open(self._iteration_log_path, "r") as f:
                self.iteration_log = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded iteration log with {len(self.iteration_log)} entries")

    def _save_iteration_log(self):
        """Write the per-iteration log to iteration_log.json."""
        with open(self._iteration_log_path, "w") as f:
            json.dump(self.iteration_log, f)

    def _increment_visit(self, node):
        """Bump the visit count, stamping the current iteration on the first visit.

        Args:
            node: Node being visited.
        """
        if node.visit_count == 0:
            node.first_visit_at_iteration = self.current_iteration
        node.visit_count += 1

    def widening_condition(self, node):
        """Return True if the node may get new children (|children| < k·N^alpha).

        Always False once the node has an `expansion_crashed` sentinel child.

        Args:
            node: Node to test.
        """

        # Crashed-expansion sentinel: if a previous iter died inside cuRobo
        # while expanding this node and was recovered as a forced terminal
        # leaf, skip expansion forever. The sentinel is one synthetic child
        # with failure_type='expansion_crashed' (see `_handle_crashed_iter`).
        for c in node.children:
            if getattr(c, "failure_type", None) == "expansion_crashed":
                return False
        return len(node.children) < self.k * (node.visit_count ** self.alpha)
    
    def save_nodes(self, directory: str = None):
        """Write all modified nodes to disk.

        Args:
            directory: Target directory (default: `nodes_dir`).
        """
        if directory is None:
            directory = self.nodes_dir
        else:
            os.makedirs(directory, exist_ok=True)

        for node in self.nodes:
            if node.dirty:
                node.save(directory)
                node.dirty = False

    def load_nodes(self):
        """Load all saved nodes from `nodes_dir` and relink parents and children.

        Returns:
            The root node (id 0), or None if nothing was saved.
        """
        if not os.path.exists(self.nodes_dir):
            return None
        
        # Find all node files
        node_files = [f for f in os.listdir(self.nodes_dir) if f.startswith('node_') and f.endswith('.npz')]
        
        if len(node_files) == 0:
            return None
        
        print(f"Loading {len(node_files)} nodes from {self.nodes_dir}")
        
        # Load all nodes first (without parent relationships)
        node_dict = {}
        for node_file in node_files:
            file_path = os.path.join(self.nodes_dir, node_file)
            node = Node.load(file_path, self.simulator, parent=None)
            node_dict[node.id] = node
            self.nodes.append(node)
        
        # Restore parent-child relationships
        for node in self.nodes:
            # Set parent
            if node.parent_id is not None and node.parent_id in node_dict:
                node.parent = node_dict[node.parent_id]
            
            # Set children
            node.children = [node_dict[child_id] for child_id in node.children_ids if child_id in node_dict]
        
        # Find root node (node with id=0)
        root_node = None
        for node in self.nodes:
            if node.id == 0:
                root_node = node
                break
        
        print(f"Loaded {len(self.nodes)} nodes with restored relationships")
        return root_node

    def mcts_iteration(self, root_node):
        """Run one MCTS iteration: descend by UCB, expanding nodes that pass the widening test.

        Args:
            root_node: Tree root.

        Returns:
            The node where the descent stopped, for backprop.
        """
        # Crash-recovery: if a previously-interrupted iter left a frontier
        # flag on disk, resume descent from the flagged node instead of
        # bumping root again. Consume the resume state once.
        if self._resume_node_id is not None:
            resume_id = self._resume_node_id
            self._resume_node_id = None
            self._resume_iteration = None
            current_node = next((n for n in self.nodes if n.id == resume_id), None)
            if current_node is None:
                print(f"  resume target id={resume_id} missing; restarting iter from root")
                current_node = root_node
                self._increment_visit(current_node)
                self._set_frontier(current_node, self.current_iteration)
            else:
                print(f"  resuming iter {self.current_iteration + 1} at node {current_node.id} "
                      f"(depth {current_node.depth})")
                # Visit counts on the descent path were already bumped + saved
                # in the original (crashed) iter; do not re-bump.
        else:
            current_node = root_node
            self._increment_visit(current_node)
            self._set_frontier(current_node, self.current_iteration)
        print(f"iter {self.current_iteration + 1}  root_visits={root_node.visit_count}  "
              f"max_depth={self.max_depth}")

        while current_node.depth < self.max_depth:
            if self.widening_condition(current_node):
                # The frontier flag is on `current_node`, so a crash during the
                # expansion resumes here (see plan()).
                self._expand([current_node])
            child = self._select_child(current_node)
            if child is None:
                break
            current_node = child
            self._increment_visit(current_node)
            # Move the descent-frontier flag to the new current node. This is
            # the resume point if cuRobo crashes during the next expansion.
            self._set_frontier(current_node, self.current_iteration)

        # Iter ended cleanly: clear the frontier (no in-flight state).
        self._clear_frontier()
        return current_node

    def _expand(self, nodes):
        """Expand several nodes in one simulator batch and add the successful candidates as children.

        Child Q starts at Q̂(o, a), or at the edge reward when no Q̂ is loaded.

        Args:
            nodes: Nodes to expand.
        """
        from collections import Counter
        from disco.search.expansion import expand

        batches = expand(self.task, self.simulator, [n.state for n in nodes],
                         K=self.num_grasps_per_object, P=self.num_places_per_grasp,
                         B=self.total_budget)
        for node, (results, value_pcd_robot) in zip(nodes, batches):
            fails = Counter(r["failure_type"] for r in results if r["is_failed"])
            print(f"  expand node {node.id} (depth {node.depth}): "
                  f"{len(results) - sum(fails.values())}/{len(results)} succeeded  failures: "
                  f"{', '.join(f'{k}={v}' for k, v in fails.items()) or 'none'}")
            self.sim_calls_last_iter += sum(
                1 for r in results if r["failure_type"] not in NOT_SIMULATED)
            results = [r for r in results if not r["is_failed"]]
            q_values = None
            if self.q_function is not None and results:
                q_values = self._predict_q_hats(value_pcd_robot, [r["action_2x8"] for r in results])
            for i, r in enumerate(results):
                self._add_child_from_result(
                    node, r, q_init=None if q_values is None else float(q_values[i]))
        self.save_nodes()

    def _select_child(self, node, virtual_loss=None):
        """Pick the child with the highest UCB (random tie-break), or a uniform one in "random" mode.

        Args:
            node: Node whose children to pick from.
            virtual_loss: Map of node id to descents already on it this batch; each lowers UCB by 1.

        Returns:
            The chosen child, or None if there are no non-failed children.
        """
        # Failed children only exist in trees from older code (and the crash
        # sentinel); never select them.
        kids = [c for c in node.children if not c.is_failed]
        if not kids:
            return None
        if self.selection_mode == "random":
            return kids[np.random.randint(len(kids))]
        np.random.shuffle(kids)
        vl = virtual_loss or {}
        return max(kids, key=lambda c: c.UCB - vl.get(c.id, 0))

    def parallel_descents_iteration(self, root_node, n):
        """Run several MCTS iterations whose descents step down the tree together.

        Nodes needing expansion at a step share one simulator batch; virtual loss spreads the descents.

        Args:
            root_node: Tree root.
            n: Number of descents.

        Returns:
            The n nodes where the descents stopped.
        """
        print(f"iters {self.current_iteration + 1}-{self.current_iteration + n}  "
              f"root_visits={root_node.visit_count}  max_depth={self.max_depth}")
        pos, virtual_loss = [root_node] * n, {}
        for _ in range(n):
            self._increment_visit(root_node)
        active = list(range(n))
        while active:
            frontier = {pos[w].id: pos[w] for w in active
                        if pos[w].depth < self.max_depth and self.widening_condition(pos[w])}
            if frontier:
                self._expand(list(frontier.values()))
            still = []
            for w in active:
                if pos[w].depth >= self.max_depth:
                    continue
                child = self._select_child(pos[w], virtual_loss)
                if child is None:
                    continue
                pos[w] = child
                self._increment_visit(child)
                virtual_loss[child.id] = virtual_loss.get(child.id, 0) + 1
                still.append(w)
            active = still
        return pos

    def plan(
        self,
        root_node: Node,
        iterations: int = 10,
        budget=None,
        env_index: int = None,
        ):
        """Run MCTS until the tree has `iterations` iterations in total or the budget runs out.

        An iteration interrupted by a crash is closed out by making its frontier a terminal leaf.

        Args:
            root_node: Root to search from (replaced by the loaded root if a saved tree exists).
            iterations: Total iterations for the tree, counting those from earlier runs.
            budget: Shared `SimBudget`, or None for no limit.
            env_index: Env whose budget share is charged for this tree's simulator calls.
        """
        if self.loaded_root_node is not None:
            root_node = self.loaded_root_node
            print(f"Using loaded root node (id={root_node.id}) instead of provided root node")
        elif root_node not in self.nodes:
            self.nodes.append(root_node)

        self.nodes.sort(key=lambda n: n.id)
        root_node.C = self.C

        # If a previous run crashed mid-iteration, the frontier protocol
        # left a flag on disk and a sidecar with the iter number. Resume
        # at that iter rather than skipping it (root.visit_count already
        # reflects the in-flight bump).
        if self._resume_iteration is not None:
            start_iteration = self._resume_iteration
            print(f"crash-recovery: resuming iter {start_iteration} "
                  f"(root.visit_count={root_node.visit_count})")
        else:
            start_iteration = root_node.visit_count
            if start_iteration > 0:
                print(f"Resuming from iteration {start_iteration} (of {iterations})")

        # If we are resuming a crashed iter, do NOT re-run expansion at the
        # frontier (almost certainly re-crashes cuRobo on the same input).
        # Instead: mark frontier as a forced terminal leaf via Option A
        # (sentinel `expansion_crashed` child),
        # backprop the crashed iter normally (G=0 at leaf, recursion adds
        # edge rewards), advance past the crashed iter, and start fresh
        # at the next iter from the root.
        if self._resume_node_id is not None:
            self.current_iteration = start_iteration
            frontier = next(
                (n for n in self.nodes if n.id == self._resume_node_id), None,
            )
            if frontier is None:
                print(f"crash-recovery: resume target id={self._resume_node_id} "
                      f"missing; skipping recovery")
            else:
                print(f"crash-recovery: marking node {frontier.id} (depth "
                      f"{frontier.depth}) as forced terminal leaf for iter "
                      f"{start_iteration}")
                self._handle_crashed_iter(frontier)
                # Backprop the crashed iter with the frontier as its leaf.
                self.backpropagate(frontier)
                self._clear_frontier()
                self.save_nodes()
                start_iteration += 1
            # Either way, consume the in-iter resume hooks so the next
            # call to mcts_iteration starts from root normally.
            self._resume_node_id = None
            self._resume_iteration = None

        i = start_iteration
        while i < iterations:
            if budget is not None and budget.exhausted():
                print(f"simulation budget exhausted ({budget.used()}/{budget.total}); "
                      f"stopping at iter {i}")
                break

            self.current_iteration = i
            self.sim_calls_last_iter = 0
            n = min(self.parallel_descents, iterations - i)
            leaves = ([self.mcts_iteration(root_node)] if n == 1
                      else self.parallel_descents_iteration(root_node, n))

            if budget is not None:
                budget.charge(env_index, self.sim_calls_last_iter)
            for leaf in leaves:
                self.backpropagate(leaf)
            print(f"iter {i + n}/{iterations} done  leaves={[l.id for l in leaves]}  "
                  f"depth={[l.depth for l in leaves]}  "
                  f"return={[round(self.path_return(l), 3) for l in leaves]}  "
                  f"sim_calls={self.sim_calls_last_iter}")
            self.save_nodes()
            i += n

    @staticmethod
    def path_return(leaf):
        """Return the sum of edge rewards from the root to the leaf.

        Args:
            leaf: Last node on the path.
        """
        R, node = 0.0, leaf
        while node.parent is not None:
            R += node.reward
            node = node.parent
        return R

    @classmethod
    def backpropagate(cls, leaf):
        """Move Q of every node on the root-to-leaf path toward the path return (running mean).

        Visit counts are bumped during the descent, not here.

        Args:
            leaf: Node where the descent stopped.
        """
        R = cls.path_return(leaf)
        node = leaf
        while node is not None:
            if node.parent is not None:
                node.Q = node.Q + (R - node.Q) / max(node.visit_count, 1)
                node.Q_history.append(node.Q)
            node.dirty = True
            node = node.parent

    # ============================================================ #
    #                       Q-FUNCTION HELPERS
    # ============================================================ #

    # +0.615 robot-frame x-shift, matching the convention used everywhere
    # else in the codebase (action_sampling, dataset pipelines).
    _ROBOT_BASE_X_OFFSET = 0.615

    def _load_q_function(self, ckpt_path):
        """Build the value network Q̂ and load its weights.

        Args:
            ckpt_path: Path to the value-network checkpoint.

        Returns:
            The network in eval mode on `q_function_device`.
        """
        from disco.flowmatch.modeling.policy.value_network import ValueNetwork
        from disco.flowmatch.utils.common_utils import count_parameters
        from disco.envs.shelf_packing_base import load_checkpoint_for_eval

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Q-function checkpoint not found: {ckpt_path}")

        model = ValueNetwork(
            embedding_dim=120,
            num_attn_heads=8,
            num_shared_attn_layers=4,
        )
        count_parameters(model)
        for n, p in model.named_parameters():
            if p.requires_grad and p.ndim > 1 and not p.is_contiguous():
                p.data = p.contiguous()
        print(f"loading Q-function from {ckpt_path} onto {self.q_function_device}")
        model = load_checkpoint_for_eval(ckpt_path, model)
        model = model.to(self.q_function_device).eval()
        return model

    @torch.no_grad()
    def _predict_q_hats(self, parent_pcd_xyz_robot, action_2x8_list, target_obj_ids=None):
        """Predict Q̂(o, a) for a batch of candidate actions in one scene.

        Args:
            parent_pcd_xyz_robot: (N, 3) xyz scene cloud in the robot frame.
            action_2x8_list: List of (2, 8) world-frame (grasp, place) action tokens.
            target_obj_ids: Unused; kept for old callers.

        Returns:
            (M,) numpy array of predicted values, one per action.
        """
        if not action_2x8_list:
            return np.zeros((0,), dtype=np.float32)

        M = len(action_2x8_list)
        actions_world = torch.stack(action_2x8_list, dim=0).float()        # (M, 2, 8)
        actions_robot = actions_world.clone()
        actions_robot[..., 0] += self._ROBOT_BASE_X_OFFSET

        xyz = torch.from_numpy(parent_pcd_xyz_robot).float().to(self.q_function_device)  # (N, 3)
        actions_t = actions_robot.to(self.q_function_device)                # (M, 2, 8)

        # Chunk over M to avoid the (M, N, 3) self-attention OOM on a
        # 24 GB card. A one-shot call over a full expansion batch
        # tripped torch.OutOfMemoryError (~14 GiB).
        # Default chunk_size matches what comfortably fits under the
        # value-net's 4096-point attention; tunable via env var if needed.
        chunk_size = int(os.environ.get("Q_FUNCTION_CHUNK_SIZE", 4))
        out_chunks = []
        with torch.no_grad():
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                pcd_chunk = xyz.unsqueeze(0).expand(end - start, -1, -1).contiguous()
                act_chunk = actions_t[start:end]
                q_chunk = self.q_function(pcd=pcd_chunk, actions=act_chunk)
                out_chunks.append(q_chunk.detach().cpu())
        q_hats = torch.cat(out_chunks, dim=0)
        return q_hats.numpy()

    def _add_child_from_result(self, parent_node, result, q_init=None):
        """Create a child node from an expansion result and attach it to the parent.

        Args:
            parent_node: Node that was expanded.
            result: Result dict from `disco.search.expansion.expand`.
            q_init: Initial Q from Q̂, or None to start Q at the edge reward.
        """
        child = Node(
            id=len(self.nodes),
            simulator=self.simulator,
            state=result["state"].clone(),
            parent=parent_node,
            prev_action=result["action"].clone(),
            prev_trajectory=result.get("trajectory"),
            reward=float(result.get("reward", 0.0)),
            C=parent_node.C,
            is_failed=bool(result.get("is_failed", False)),
            failure_type=result.get("failure_type"),
        )
        child.created_at_iteration = self.current_iteration
        if q_init is not None and not child.is_failed:
            child.Q = float(q_init)
            child.predicted_value = float(q_init)
        parent_node.children.append(child)
        parent_node.children_ids.append(child.id)
        self.nodes.append(child)
        parent_node.dirty = True

    # ============================================================ #
    #                       CRASH RECOVERY
    # ============================================================ #

    def _write_in_flight(self, iteration: int, frontier_node_id: int):
        """Atomically record the running iteration and its frontier node in in_flight.json.

        Args:
            iteration: Current iteration index.
            frontier_node_id: Id of the node the descent is at.
        """
        if self._in_flight_path is None:
            return
        tmp = self._in_flight_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"iteration": iteration, "frontier_node_id": frontier_node_id}, f)
        os.replace(tmp, self._in_flight_path)  # atomic on POSIX

    def _clear_in_flight(self):
        """Delete the in-flight sidecar file, if present."""
        if self._in_flight_path is not None and os.path.exists(self._in_flight_path):
            os.remove(self._in_flight_path)

    def _read_in_flight(self):
        """Return the in-flight sidecar contents, or None if missing or unreadable."""
        if self._in_flight_path is None or not os.path.exists(self._in_flight_path):
            return None
        try:
            with open(self._in_flight_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _set_frontier(self, node, iteration: int):
        """Mark the node as the only descent frontier and persist it for crash recovery.

        Args:
            node: Node the descent is currently at.
            iteration: Current iteration index.
        """
        for n in self.nodes:
            if n.is_expansion_frontier and n is not node:
                n.is_expansion_frontier = False
                n.dirty = True
        if not node.is_expansion_frontier:
            node.is_expansion_frontier = True
            node.dirty = True
        self._write_in_flight(iteration, node.id)
        self.save_nodes()

    def _clear_frontier(self):
        """Clear all frontier flags and remove the in-flight sidecar."""
        for n in self.nodes:
            if n.is_expansion_frontier:
                n.is_expansion_frontier = False
                n.dirty = True
        self.save_nodes()
        self._clear_in_flight()

    def _handle_crashed_iter(self, frontier):
        """Turn a node whose expansion crashed into a terminal leaf.

        Adds one failed `expansion_crashed` sentinel child, which blocks further widening.

        Args:
            frontier: Node whose expansion crashed.

        Returns:
            The sentinel child (the existing one if already present).
        """
        # If a sentinel already exists (we crashed twice on the same node),
        # don't add another.
        for c in frontier.children:
            if getattr(c, "failure_type", None) == "expansion_crashed":
                return c

        sentinel = Node(
            id=len(self.nodes),
            simulator=self.simulator,
            state=frontier.state.clone(),
            parent=frontier,
            prev_action=None,
            prev_trajectory=None,
            reward=0.0,
            C=frontier.C,
            is_failed=True,
            failure_type="expansion_crashed",
        )
        sentinel.created_at_iteration = self.current_iteration
        frontier.children.append(sentinel)
        frontier.children_ids.append(sentinel.id)
        self.nodes.append(sentinel)
        frontier.dirty = True
        return sentinel

    def _detect_resume_state(self):
        """Set the resume node and iteration if a previous run was interrupted mid-iteration."""
        sidecar = self._read_in_flight()
        flag_holders = [n for n in self.nodes if n.is_expansion_frontier]

        if sidecar is None and not flag_holders:
            return

        if sidecar is not None:
            self._resume_iteration = int(sidecar.get("iteration", -1))
            sidecar_id = int(sidecar.get("frontier_node_id", -1))
            chosen_id = sidecar_id
        else:
            chosen_id = None
            self._resume_iteration = None

        if flag_holders:
            ids = [n.id for n in flag_holders]
            if chosen_id is None:
                # No sidecar but flags on disk — pick the deepest flagged node
                # as the resume point (most recent in descent).
                deepest = max(flag_holders, key=lambda n: n.depth)
                chosen_id = deepest.id
                # iteration is unknown; will be derived from root.visit_count - 1.
            elif chosen_id not in ids:
                print(f"[resume] sidecar frontier_node_id={chosen_id} not flagged; "
                      f"falling back to deepest flagged node from {ids}")
                deepest = max(flag_holders, key=lambda n: n.depth)
                chosen_id = deepest.id
            if len(flag_holders) > 1:
                print(f"[resume] WARNING: multiple flagged nodes on disk: {ids}. "
                      f"Resuming at id={chosen_id}, clearing the rest.")

        self._resume_node_id = chosen_id
        print(f"[resume] in-flight iteration detected: iter={self._resume_iteration} "
              f"frontier_node_id={self._resume_node_id}")
