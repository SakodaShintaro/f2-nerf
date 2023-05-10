"""
Match two trajectories about translation, scaling and rotation.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_trajectory_A', type=str)
    parser.add_argument('path_to_trajectory_B', type=str)
    return parser.parse_args()


def calc_mat_A_to_B(traj_A: np.array, traj_B: np.array):
    axis_convert_mat_B_to_A = np.array(
        [[0, -1,  0,  0],
         [0,  0, -1,  0],
         [1,  0,  0,  0],
         [0,  0,  0,  1]], dtype=np.float64
    )
    axis_convert_mat_A_to_B = axis_convert_mat_B_to_A.T

    # Replace the axis.
    traj_A = np.hstack((traj_A, np.ones((traj_A.shape[0], 1))))
    traj_A = np.dot(axis_convert_mat_A_to_B, traj_A.T).T
    traj_A = traj_A[:, 0:3]

    # Make the origin the center.
    center_A = traj_A[0].copy()
    center_B = traj_B[0].copy()
    traj_A -= center_A
    traj_B -= center_B

    # set scale
    s2_A = (traj_A ** 2.).sum()
    s2_B = (traj_B ** 2.).sum()

    # centred Frobenius norm
    norm_A = np.sqrt(s2_A)
    norm_B = np.sqrt(s2_B)
    print(norm_A, norm_B)

    # scale to equal (unit) norm
    traj_A /= norm_A
    traj_B /= norm_B

    # Align rotation and scale.
    rotation_matrix, _ = orthogonal_procrustes(traj_A, traj_B)

    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[0:3, 0:3] = rotation_matrix.T

    scaling_factor = norm_B / norm_A
    scaling_matrix_4x4 = np.eye(4)
    scaling_matrix_4x4[0:3, 0:3] = scaling_factor * np.eye(3)

    translation_vec = center_A - center_B
    translation_matrix_4x4 = np.eye(4)
    translation_matrix_4x4[0:3, 3] = -translation_vec

    # Composition of matrices.
    mat = np.eye(4)
    mat = np.dot(translation_matrix_4x4, mat)
    mat = np.dot(scaling_matrix_4x4, mat)
    mat = np.dot(rotation_matrix_4x4, mat)
    return axis_convert_mat_A_to_B, mat, scaling_factor


def invert_affine_transform(matrix, scaling_factor):
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    R_inv = R.T

    t_inv = -R_inv @ t

    matrix_inv = np.eye(4)
    matrix_inv[:3, :3] = R_inv / scaling_factor / scaling_factor
    matrix_inv[:3, 3] = t_inv / scaling_factor / scaling_factor

    return matrix_inv


def apply_mat(mat, vec):
    result = np.hstack((vec, np.ones((vec.shape[0], 1))))
    result = np.dot(mat, result.T).T
    result = result[:, 0:3]
    return result


if __name__ == "__main__":
    args = parse_args()
    path_to_trajectory_A = args.path_to_trajectory_A
    path_to_trajectory_B = args.path_to_trajectory_B

    npy_A = np.load(path_to_trajectory_A)
    pose = npy_A[:, 0:12]
    pose = pose.reshape(-1, 3, 4)
    traj_A = pose[:, 0:3, 3]

    df_B = pd.read_csv(path_to_trajectory_B, sep='\t', index_col=0)
    traj_B = df_B[['x', 'y', 'z']].values

    min_length = min(traj_A.shape[0], traj_B.shape[0])
    traj_A = traj_A[:min_length]
    traj_B = traj_B[:min_length]

    print(traj_A[0], traj_B[0])

    mat_axis_convert_c2m, mat_proc_c2m, scaling_factor = calc_mat_A_to_B(
        traj_A.copy(), traj_B.copy())
    mat_c2m = np.dot(mat_proc_c2m, mat_axis_convert_c2m)

    mat_proc_m2c = invert_affine_transform(mat_proc_c2m, scaling_factor)
    mat_axis_convert_m2c = mat_axis_convert_c2m.T
    mat_m2c = np.dot(mat_axis_convert_m2c, mat_proc_m2c)

    # Fixed decimal point representation
    np.set_printoptions(precision=6, suppress=True)
    print("mat_c2m")
    print(mat_c2m)
    print("mat_m2c")
    print(mat_m2c)

    # apply
    traj_A_converted = apply_mat(mat_c2m, traj_A.copy())
    traj_B_converted = apply_mat(mat_m2c, traj_B.copy())

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # Upper left: A alone
    # Upper right: B alone
    # Lower left: A and B converted to A coordinate system
    # Lower right: B and A converted to B coordinate system
    axes[0, 0].plot(traj_A[:, 2], traj_A[:, 0])
    axes[0, 0].set_title('A')
    axes[0, 1].plot(traj_B[:, 0], traj_B[:, 1])
    axes[0, 1].set_title('B')
    axes[1, 0].plot(traj_A[:, 2], traj_A[:, 0], label='A')
    axes[1, 0].plot(traj_B_converted[:, 2],
                    traj_B_converted[:, 0], label='B')
    axes[1, 0].set_title('B to A')
    axes[1, 0].legend()
    axes[1, 1].plot(traj_B[:, 0], traj_B[:, 1], label='B')
    axes[1, 1].plot(traj_A_converted[:, 0],
                    traj_A_converted[:, 1], label='A')
    axes[1, 1].set_title('A to B')
    axes[1, 1].legend()
    axes[2, 0].plot(traj_A[:, 2], traj_A[:, 1], label='A')
    axes[2, 0].plot(traj_B_converted[:, 2],
                    traj_B_converted[:, 1], label='B')
    axes[2, 0].set_title('B to A')
    axes[2, 0].legend()
    axes[2, 1].plot(traj_B[:, 0], traj_B[:, 2], label='B')
    axes[2, 1].plot(traj_A_converted[:, 0],
                    traj_A_converted[:, 2], label='A')
    axes[2, 1].set_title('A to B')
    axes[2, 1].legend()

    axes[0, 0].set_aspect('equal')
    axes[0, 1].set_aspect('equal')
    axes[1, 0].set_aspect('equal')
    axes[1, 1].set_aspect('equal')
    axes[2, 0].set_aspect('equal')
    axes[2, 1].set_aspect('equal')

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    save_path = 'compare_trajectory.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    print(f'save to {save_path}')
