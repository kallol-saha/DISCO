"""Per-env stacked / tower checks on ObjectStacking (GPU sim, num_envs=32)."""
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env():
    import gymnasium as gym
    from disco.envs.object_stacking_env import ObjectStacking  # noqa: F401  (registers env)
    e = gym.make("ObjectStacking-v1", num_envs=32, obs_mode="state", control_mode="pd_joint_pos",
                 render_mode=None, env_path="assets/environments/object_stacking/test", env_index=5)
    e.reset(seed=[0] * 32)
    yield e.unwrapped


def test_stack_only_in_one_slot(env):
    poses = env.object_poses_tensor.clone()            # (32, n, 7)
    dz0 = env.get_object_current_dims(env.object_names[0])[2]
    dz1 = env.get_object_current_dims(env.object_names[1])[2]
    poses[3, 1, :2] = poses[3, 0, :2]                   # object 1 above object 0, slot 3 only
    poses[3, 1, 2] = poses[3, 0, 2] + dz0 / 2 + dz1 / 2 + 0.01
    env.set_object_poses_tensor(poses)
    env.wait_for_stability(max_steps=200)

    S = env.stacked_matrix()
    assert S.shape == (32, env.num_objects, env.num_objects)
    assert bool(S[3, 1, 0])
    assert not bool(S[0, 1, 0])

    h3, chain3 = env.tallest_tower(3)
    h0, _ = env.tallest_tower(0)
    assert h3 >= 2 and chain3[:2] == [0, 1]
    assert h0 == 1

    th = env.evaluate()["tower_height"]
    assert th.shape == (32,) and th[3] >= 2 and th[0] == 1
    assert env.is_stacked_on(1, 0, env_idx=3) and not env.is_stacked_on(1, 0, env_idx=0)
