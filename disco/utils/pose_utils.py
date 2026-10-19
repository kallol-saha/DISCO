"""
7-vector pose helpers (x, y, z, qw, qx, qy, qz) and stacking face-down orientations.
"""
import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat, qconjugate, qmult, quat2mat

# All quaternions are SAPIEN convention: wxyz, scalar-first.


def quat_from_axis_angle(axis, angle):
    """Return the wxyz quaternion for a rotation about an axis.

    Args:
        axis: 3D rotation axis.
        angle: Rotation angle in radians.
    """
    return mat2quat(axangle2mat(np.asarray(axis, dtype=np.float64), float(angle)))


def quat_mul(q1, q2):
    """Return the product q1 * q2 of two wxyz quaternions.

    Args:
        q1: Left wxyz quaternion.
        q2: Right wxyz quaternion.
    """
    return qmult(q1, q2)


def quat_rotate(q, v):
    """Rotate a 3D vector by a wxyz quaternion.

    Args:
        q: wxyz quaternion.
        v: 3D vector.
    """
    return quat2mat(np.asarray(q, dtype=np.float64)) @ np.asarray(v, dtype=np.float64)


def inv_pose(pose):
    """Invert a 7-vec pose (x, y, z, qw, qx, qy, qz).

    Args:
        pose: 7-vec pose.
    """
    pose = np.asarray(pose, dtype=np.float64)
    pos, q = pose[:3], pose[3:]
    inv_q = qconjugate(q)  # unit quat inverse == conjugate
    inv_pos = -quat_rotate(inv_q, pos)
    return np.concatenate([inv_pos, inv_q])


def compose_pose(a, b):
    """Compose two 7-vec poses, mapping b from a's frame into the world.

    Args:
        a: Pose of frame a in the world.
        b: Pose expressed in frame a.

    Returns:
        7-vec pose of b in the world.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pos = a[:3] + quat_rotate(a[3:], b[:3])
    q = quat_mul(a[3:], b[3:])
    return np.concatenate([pos, q])


def quat_angle(qa, qb):
    """Return the angle in radians between two wxyz quaternions, ignoring sign.

    Args:
        qa: First wxyz quaternion.
        qb: Second wxyz quaternion.
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    d = abs(float(np.dot(qa, qb)))
    return 2.0 * np.arccos(min(1.0, d))


# Six face-down orientations (which face of the moved object's local frame
# becomes the down-facing one in world after placement). The base's yaw is
# multiplied on top of this so the moved object's in-plane orientation
# matches the base.
FACE_QUATS = np.stack([
    quat_from_axis_angle((1, 0, 0), np.pi),       # 0: +z face down
    quat_from_axis_angle((1, 0, 0), 0.0),         # 1: -z face down (natural upright)
    quat_from_axis_angle((0, 1, 0), -np.pi / 2),  # 2: +x face down
    quat_from_axis_angle((0, 1, 0),  np.pi / 2),  # 3: -x face down
    quat_from_axis_angle((1, 0, 0),  np.pi / 2),  # 4: +y face down
    quat_from_axis_angle((1, 0, 0), -np.pi / 2),  # 5: -y face down
])

