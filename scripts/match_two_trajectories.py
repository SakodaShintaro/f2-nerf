"""
Match two trajectories about translation, scaling and rotation.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation
from convert_pose_tsv_to_f2_format import AXIS_CONVERT_MAT1, AXIS_CONVERT_MAT2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_trajectory_A', type=str)
    parser.add_argument('path_to_trajectory_B', type=str)
    return parser.parse_args()


def calc_mat_B2A(traj_A: np.array, traj_B: np.array):
    # Replace the axis.
    traj_B = apply_mat(np.eye(4), traj_B)

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
    rotation_matrix, _ = orthogonal_procrustes(traj_B, traj_A)

    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[0:3, 0:3] = rotation_matrix.T

    scaling_factor = norm_A / norm_B
    scaling_matrix_4x4 = np.eye(4)
    scaling_matrix_4x4[0:3, 0:3] = scaling_factor * np.eye(3)

    translation_vec_B = center_B
    translation_matrix_4x4_B = np.eye(4)
    translation_matrix_4x4_B[0:3, 3] = -translation_vec_B
    translation_vec_A = center_A
    translation_matrix_4x4_A = np.eye(4)
    translation_matrix_4x4_A[0:3, 3] = translation_vec_A

    # Composition of matrices.
    mat = np.eye(4)
    mat = np.dot(translation_matrix_4x4_B, mat)
    mat = np.dot(scaling_matrix_4x4, mat)
    mat = np.dot(rotation_matrix_4x4, mat)
    mat = np.dot(translation_matrix_4x4_A, mat)
    return mat, scaling_factor


def apply_mat(mat, vec):
    result = np.hstack((vec, np.ones((vec.shape[0], 1))))
    # result = result @ AXIS_CONVERT_MAT1
    result = result.T
    result = AXIS_CONVERT_MAT2 @ result
    result = mat @ result
    result = result.T
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

    mat_B2A, scale = calc_mat_B2A(traj_A.copy(), traj_B.copy())
    mat_A2B = np.linalg.inv(mat_B2A)

    # Fixed decimal point representation
    np.set_printoptions(precision=6, suppress=True)
    print("mat_A2B")
    print(mat_A2B)
    print("mat_B2A")
    print(mat_B2A)

    # apply
    traj_A_converted = apply_mat(mat_A2B, traj_A.copy())
    traj_B_converted = apply_mat(mat_B2A, traj_B.copy())

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

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
    axes[0, 0].set_aspect('equal')
    axes[0, 1].set_aspect('equal')
    axes[1, 0].set_aspect('equal')
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()

    def plot_arrow(ax, pose_mat, initial_dir, position, color):
        if initial_dir == "x":
            vec1 = pose_mat @ np.array([1, -0.25, 0])
            vec2 = pose_mat @ np.array([1, +0.25, 0])
            ax.arrow(position[0], position[1],
                     vec1[0], vec1[1], color=color, width=0.1)
            ax.arrow(position[0], position[1],
                     vec2[0], vec2[1], color=color, width=0.1)
        else:
            vec1 = pose_mat @ np.array([-0.25, 0, 1])
            vec2 = pose_mat @ np.array([+0.25, 0, 1])
            ax.arrow(position[2], position[0],
                     vec1[2], vec1[0], color=color, width=0.1)
            ax.arrow(position[2], position[0],
                     vec2[2], vec2[0], color=color, width=0.1)

    n = min_length
    for i in range(0, n, n // 6):
        orientation_A = pose[i, 0:3, 0:3]
        quat_B = df_B.iloc[i, 3:7].values
        orientation_B = Rotation.from_quat(quat_B).as_matrix()

        plot_arrow(axes[0, 0], orientation_A, "z", traj_A[i, 0:3], "red")
        plot_arrow(axes[0, 1], orientation_B, "x", traj_B[i, 0:3], "red")

        # calc converted_B
        converted_B = mat_B2A[0:3, 0:3] @ \
            AXIS_CONVERT_MAT2[0:3, 0:3] @ \
            orientation_B  @ \
            AXIS_CONVERT_MAT1[0:3, 0:3]
        plot_arrow(axes[1, 0], orientation_A, "z", traj_A[i, 0:3], "red")
        plot_arrow(axes[1, 0], converted_B, "z",
                   traj_B_converted[i, 0:3], "blue")

        # calc converted_A
        # converted_A = axis_convert_mat_A_to_B[0:3, 0:3] @ \
        #     orientation_A @ \
        #     Rotation.from_euler("zyx", [0, +90, 0], degrees=True).as_matrix()
        plot_arrow(axes[1, 1], orientation_B, "x", traj_B[i, 0:3], "red")
        # plot_arrow(axes[1, 1], converted_A, "x",
        #            traj_A_converted[i, 0:3], "blue")

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    save_path = 'compare_trajectory.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    print(f'save to {save_path}')
