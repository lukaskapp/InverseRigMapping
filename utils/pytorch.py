"""
-----------------------------------------------------------------------------
This file has been developed within the scope of the
Technical Director course at Filmakademie Baden-Wuerttemberg.
http://technicaldirector.de

Written by Lukas Kapp
Copyright (c) 2023 Animationsinstitut of Filmakademie Baden-Wuerttemberg
-----------------------------------------------------------------------------
"""

import torch
import numpy as np

def normalize_tensor(tensor, min_val, max_val, mean_val):
    eps = 1e-7
    return 2 * (tensor - mean_val) / (max_val - min_val + eps)


def denormalize_tensor(tensor, min_val, max_val, mean_val):
    return tensor * (max_val - min_val) / 2 + mean_val


def calculate_min_max_mean(tensor):
    min_val, _ = torch.min(tensor, dim=0, keepdim=True)
    max_val, _ = torch.max(tensor, dim=0, keepdim=True)
    mean_val = torch.mean(tensor, dim=0, keepdim=True)

    return min_val, max_val, mean_val


def rotation_matrix_to_quaternion(rot_matrix):
    trace = torch.trace(rot_matrix)
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        quat_w = 0.25 * s
        quat_x = (rot_matrix[2,1] - rot_matrix[1,2]) / s
        quat_y = (rot_matrix[0,2] - rot_matrix[2,0]) / s
        quat_z = (rot_matrix[1,0] - rot_matrix[0,1]) / s
    else:
        diagonal_elements = torch.tensor([rot_matrix[0,0], rot_matrix[1,1], rot_matrix[2,2]])
        max_index = torch.argmax(diagonal_elements).item()
        if max_index == 0:
            s = torch.sqrt(1.0 + rot_matrix[0,0] - rot_matrix[1,1] - rot_matrix[2,2]) * 2
            quat_w = (rot_matrix[2,1] - rot_matrix[1,2]) / s
            quat_x = 0.25 * s
            quat_y = (rot_matrix[0,1] + rot_matrix[1,0]) / s
            quat_z = (rot_matrix[0,2] + rot_matrix[2,0]) / s
        elif max_index == 1:
            s = torch.sqrt(1.0 + rot_matrix[1,1] - rot_matrix[0,0] - rot_matrix[2,2]) * 2
            quat_w = (rot_matrix[0,2] - rot_matrix[2,0]) / s
            quat_x = (rot_matrix[0,1] + rot_matrix[1,0]) / s
            quat_y = 0.25 * s
            quat_z = (rot_matrix[1,2] + rot_matrix[2,1]) / s
        else:
            s = torch.sqrt(1.0 + rot_matrix[2,2] - rot_matrix[0,0] - rot_matrix[1,1]) * 2
            quat_w = (rot_matrix[1,0] - rot_matrix[0,1]) / s
            quat_x = (rot_matrix[0,2] + rot_matrix[2,0]) / s
            quat_y = (rot_matrix[1,2] + rot_matrix[2,1]) / s
            quat_z = 0.25 * s
    
    quaternion = torch.tensor([quat_w, quat_x, quat_y, quat_z])
    
    return quaternion


def batch_rotation_matrix_to_quaternion(rot_matrices):
    quaternions = []
    for i in range(rot_matrices.shape[0]):
        q = rotation_matrix_to_quaternion(rot_matrices[i])
        quaternions.append(q)
    return torch.stack(quaternions)


def quaternion_to_rotation_matrix(quat):
    # Normalize the quaternion to unit length
    quat = quat / torch.sqrt(torch.sum(quat**2))

    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Compute the rotation matrix elements
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - x*w)
    
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x**2 + y**2)

    rotation_matrix = torch.tensor([[r00, r01, r02],
                                    [r10, r11, r12],
                                    [r20, r21, r22]])

    return rotation_matrix


def batch_quaternion_to_rotation_matrix(quaternions):
    rotation_matrices = []
    for i in range(quaternions.shape[0]):
        R = quaternion_to_rotation_matrix(quaternions[i])
        rotation_matrices.append(R)
    return torch.stack(rotation_matrices)


def matrix_to_6d(rot_mat):
    # Use the first two columns of the rotation matrix to get the 6D representation
    return rot_mat[:, :2].reshape(-1)


def _6d_to_matrix(rot_6d):
    # Reshape the 6D representation back to a 3x2 matrix
    mat = rot_6d.view(-1, 3, 2)

    # Calculate the third column of the rotation matrix as the cross product of the first two columns
    third_col = torch.cross(mat[:, :, 0], mat[:, :, 1]).unsqueeze(2)

    # Construct the full rotation matrix
    return torch.cat((mat, third_col), dim=2)
