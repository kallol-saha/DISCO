"""
Holds the object stacking grasp and place policies for MCTS expansion.
"""
from disco.policy import models as M
from disco.tasks import StackingTask


class StackingPolicySampler:
    """Holder for the stacking grasp and place policies used by MCTS expansion."""

    def __init__(self, env):
        """Create an empty policy holder.

        Args:
            env: ObjectStacking simulator the policies act in.
        """
        self.env = env
        self.models = {}

    def load_policy(self, policy_type, model_path, device="cuda:0"):
        """Load a grasp or place policy checkpoint into `self.models`.

        Args:
            policy_type: "grasp_only" or "place_conditional" (same names as the shelf sampler).
            model_path: Policy checkpoint path.
            device: Device to load the policy onto.
        """
        kind = {"grasp_only": "grasp", "place_conditional": "place"}[policy_type]
        model = M.build_policy(kind, StackingTask.policy_channels)
        self.models[kind] = M.load_weights(model, model_path).to(device).eval()
