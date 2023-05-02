""" A script to plot camera metadata
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_camera_meta_npy", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_to_camera_meta_npy = args.path_to_camera_meta_npy

    camera_meta = np.load(path_to_camera_meta_npy)
    print(camera_meta.shape)
    poses = camera_meta[:, 0:12]
    poses = poses.reshape(-1, 3, 4)
    poses[:, :, 3] -= poses[0, :,  3]

    rear_pos = poses[:, :, 3]

    default_pose_axle = np.array([0, 0, 0, 1])
    default_pose_front = np.array([0, 0, -1, 1])

    base_dir = os.path.dirname(path_to_camera_meta_npy)
    save_dir = f"{base_dir}/camera_pose"
    os.makedirs(save_dir, exist_ok=True)

    for i, pose in enumerate(tqdm(poses)):
        mat = np.eye(4)
        mat[0:3, 0:4] = pose
        curr_axle = np.dot(mat, default_pose_axle)
        curr_front = np.dot(mat, default_pose_front)
        plt.plot(rear_pos[:i + 1, 2], rear_pos[:i + 1, 0])
        plt.plot(curr_axle[2], curr_axle[0])
        plt.arrow(curr_axle[2],
                  curr_axle[0],
                  curr_front[2] - curr_axle[2],
                  curr_front[0] - curr_axle[0],
                  head_width=0.2,
                  head_length=0.4,
                  color="red")
        plt.axis("equal")

        # flip y axis
        plt.ylim(plt.ylim()[::-1])

        plt.xlabel("z")
        plt.ylabel("x")
        save_path = f"{save_dir}/{i:08d}.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
