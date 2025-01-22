import torch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

def normal_so3(batch_size, scale=1.0):
    """
    Generate a batch of random SO(3) rotation matrices using axis/angle representation.


    :param batch_size: Number of rotation matrices to generate.
    :param scale: The maximum angle in radians by which to rotate.
    :return: A tensor of shape (batch_size, 3, 3) containing SO(3) rotation matrices.
    """

    v = torch.randn(batch_size, 3) * scale
    rotation_matrices = exp_map(v)
    return rotation_matrices


def random_so3(batch_size):
    """
    Generate a batch of random SO(3) rotation matrices using quaternions.

    :param batch_size: Number of rotation matrices to generate.
    :return: A tensor of shape (batch_size, 3, 3) containing SO(3) rotation matrices.
    """
    # Generate random quaternions
    q = torch.randn(batch_size, 4)
    q = torch.nn.functional.normalize(q, p=2, dim=1)

    # Convert quaternions to rotation matrices
    rotation_matrices = torch.zeros(batch_size, 3, 3)

    # Quaternion components
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # First row of the rotation matrix
    rotation_matrices[:, 0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
    rotation_matrices[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    rotation_matrices[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    rotation_matrices[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    rotation_matrices[:, 1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
    rotation_matrices[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    rotation_matrices[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    rotation_matrices[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    rotation_matrices[:, 2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)

    return rotation_matrices


def exp_map(v):
    return quaternion_to_matrix(axis_angle_to_quaternion(v))

def log_map(r):
    '''
    Given a matrix r in SO(3), compute the log map.
    Input:
        r: Bx3x3
    Output:
        log_r: Bx3
    '''
    return quaternion_to_axis_angle(matrix_to_quaternion(r))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles



def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret




def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def se3_inverse(p, r):
    r_inv = r.transpose(-1, -2)
    p_inv = -torch.matmul(r_inv, p.unsqueeze(-1)).squeeze(-1)
    return p_inv, r_inv


def apply_transform(p, r, p0, r0):
    p1 = torch.matmul(r, p0.unsqueeze(-1)).squeeze(-1) + p
    r1 = torch.matmul(r, r0)
    return p1, r1

if __name__ == '__main__':

    B = 100
    r = random_so3(B)

    log_r = log_map(r)
    r2 = exp_map(log_r)

    print(torch.allclose(r, r2, atol=1e-5))


def se3_from_rot_pos(rot, pos):
    """
    Convert rotations and positions to SE(3) matrices.

    Args:
        rot: rotations as rotation matrices, as tensor of shape (..., 3, 3).
        pos: positions as translation vectors, as tensor of shape (..., 3).

    Returns:
        SE(3) matrices as tensor of shape (..., 4, 4).
    """
    H = torch.eye(4, device=rot.device, dtype=rot.dtype).expand(rot.shape[:-2] + (4, 4)).clone()
    H[..., :3, :3] = rot
    H[..., :3, 3] = pos
    return H

def rot_pos_from_se3(H):
    """
    Convert SE(3) matrices to rotations and positions.

    Args:
        H: SE(3) matrices, as tensor of shape (..., 4, 4).

    Returns:
        Tuple of rotations as rotation matrices, as tensor of shape (..., 3, 3),
        and positions as translation vectors, as tensor of shape (..., 3).
    """
    return H[..., :3, :3], H[..., :3, 3]

def random_se3(batch_size):
    '''
    This function creates SE(3) homogeneous matrices
    :param batch_size: N
    :return: Nx4x4 tensor of shape (batch_size, 4, 4)
    '''
    H = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    H[:, :3, :3] = random_so3(batch_size)
    H[:, :3, 3] = torch.randn(batch_size, 3)
    return H

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    dtype = torch.float32
    rot = torch.randn(2, 3, 3, device=device, dtype=dtype)
    pos = torch.randn(2, 3, device=device, dtype=dtype)
    se3 = se3_from_rot_pos(rot, pos)
    assert se3.shape == (2, 4, 4)
    assert se3.device == rot.device
    assert se3.dtype == rot.dtype
    assert torch.allclose(se3[..., :3, :3], rot)
    assert torch.allclose(se3[..., :3, 3], pos)


if __name__ == "__main__":
    test()
    print("Passed")