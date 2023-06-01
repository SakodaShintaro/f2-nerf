""" Modified offset by particles_log.
"""

import argparse
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R
import yaml
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str, help='Path to the log file')
    return parser.parse_args()


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    args = parse_args()
    log_file = args.log_file
    log_dir = os.path.dirname(log_file)
    save_dir = f"{log_dir}/../particles_plot"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(file_path)
    target_yaml = f"{base_dir}/../src/ros2-f2-nerf/config/parameters.yaml"
    with open(target_yaml) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # get weight range
    df = pd.read_csv(log_file, sep="\t")
    weights = df["weight"].values

    # get trajectory
    positions = df[["m03", "m13", "m23"]].values
    first_pose = df.values[0][0:12].reshape(3, 4)
    best_index = weights.argmax()
    print(f"best index: {best_index}")
    best_pose = df.values[best_index][0:12].reshape(3, 4)
    best_score = df.values[best_index][12]
    prev_score = df.values[0][12]
    print(f"score: {prev_score} -> {best_score}")
    diff_position = best_pose[0:3, 3] - first_pose[0:3, 3]
    diff_rotation = best_pose[0:3, 0:3] @ first_pose[0:3, 0:3].T
    diff_euler = R.from_matrix(diff_rotation).as_euler('xyz', degrees=True)
    print("diff")
    print(diff_position)
    print(diff_euler)

    # Change the offset in parameters.yaml
    target_dict = params["nerf_based_localizer"]["ros__parameters"]

    print("next")
    curr_position = np.array([
        target_dict["offset_position_x"],
        target_dict["offset_position_y"],
        target_dict["offset_position_z"]
    ])
    next_position = curr_position + diff_position

    curr_quat = np.array([
        target_dict["offset_rotation_x"],
        target_dict["offset_rotation_y"],
        target_dict["offset_rotation_z"],
        target_dict["offset_rotation_w"]
    ])
    curr_quat = R.from_quat(curr_quat)
    next_mat = (diff_rotation @ curr_quat.as_matrix())
    next_quat = R.from_matrix(next_mat).as_quat()

    print(f"    offset_position_x: {next_position[0]:.6f}")
    print(f"    offset_position_y: {next_position[1]:.6f}")
    print(f"    offset_position_z: {next_position[2]:.6f}")
    print(f"    offset_rotation_w: {next_quat[3]:.6f}")
    print(f"    offset_rotation_x: {next_quat[0]:.6f}")
    print(f"    offset_rotation_y: {next_quat[1]:.6f}")
    print(f"    offset_rotation_z: {next_quat[2]:.6f}")
