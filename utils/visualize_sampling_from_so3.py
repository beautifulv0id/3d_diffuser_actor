import torch
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import matplotlib.pyplot as plt
from utils.so3_util import normal_so3



def plot_orientation(positions, rotations, ax, scale=1.0, main_pose=False):
    """
    Plot a 3D orientation defined by a position and rotation matrix.

    Args:
        position (torch.Tensor): A tensor of shape (3,) representing the 3D position.
        rotation_matrix (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.
    """
    # Ensure input types are tensors

    # Create figure and 3D axis
    if main_pose:
        colors = ['r', 'g', 'b']
    else:
        colors = ['gray', 'gray', 'gray']

    # Plot origin axes
    for position, rotation in zip(positions, rotations):
        ax.quiver(
            position[0], position[1], position[2],  # Start point
            rotation[0, 0], rotation[1, 0], rotation[2, 0],  # X-axis (Red)
            color=colors[0], length=scale, normalize=True, label='X-axis (Red)'
        )

        ax.quiver(
            position[0], position[1], position[2],
            rotation[0, 1], rotation[1, 1], rotation[2, 1],  # Y-axis (Green)
            color=colors[1], length=scale, normalize=True, label='Y-axis (Green)'
        )

        ax.quiver(
            position[0], position[1], position[2],
            rotation[0, 2], rotation[1, 2], rotation[2, 2],  # Z-axis (Blue)
            color=colors[2], length=scale, normalize=True, label='Z-axis (Blue)'
        )

def sample_quaternions(batch_size: int, scale: float = 1.0) -> torch.Tensor:
    q = torch.randn(batch_size, 4) * scale
    return torch.nn.functional.normalize(q, p=2, dim=1)

def add_pos_noise(gripper_history, pos_noise_scale):
    noise = torch.randn_like(gripper_history[...,:3]) * pos_noise_scale
    return gripper_history[...,:3] + noise

def add_rot_noise(q, rot_noise_scale, quaternion_format='xyzw'):
    noise = normal_so3(q.shape[0], rot_noise_scale)
    if quaternion_format == 'xyzw':
        q = q[..., (3, 0, 1, 2)]
    rot = quaternion_to_matrix(q)
    rot = torch.bmm(rot, noise)
    q = matrix_to_quaternion(rot)
    if quaternion_format == 'xyzw':
        q = q[..., (1, 2, 3, 0)]
    return q


def main():
    tgt_pos = torch.tensor([[0.0, 0.0, 0.0]])
    tgt_q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])    
    tgt_rot = quaternion_to_matrix(tgt_q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_orientation(tgt_pos, tgt_rot, ax, scale=0.05, main_pose=True)

    sample_pos = []
    sample_rot = []
    for i in range(10):
        p_ = add_pos_noise(tgt_pos, 0.0001)
        q_ = add_rot_noise(tgt_q, 0.2, quaternion_format='xyzw')
        r_ = quaternion_to_matrix(q_)
        sample_pos.append(p_)
        sample_rot.append(r_)

    sample_pos = torch.cat(sample_pos)
    sample_rot = torch.cat(sample_rot)

    plot_orientation(sample_pos, sample_rot, ax, scale=0.05, main_pose=True)

    ax.set_xlim([tgt_pos[0][0] - 0.1, tgt_pos[0][0] + 0.1])
    ax.set_ylim([tgt_pos[0][1] - 0.1, tgt_pos[0][1] + 0.1])
    ax.set_zlim([tgt_pos[0][2] - 0.1, tgt_pos[0][2] + 0.1])

    plt.show()

if __name__ == '__main__':
    main()