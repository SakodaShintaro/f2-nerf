import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os
import yaml
from pathlib import Path
from glob import glob
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_pose_tsv", type=str)
    return parser.parse_args()


def load_camera_info_from_yaml(filename):
    with open(filename, "r") as input_file:
        camera_info_dict = yaml.safe_load(input_file)
        camera_info_dict["D"] = np.array(camera_info_dict["D"])
        camera_info_dict["K"] = np.array(camera_info_dict["K"]).reshape((3, 3))
        camera_info_dict["R"] = np.array(camera_info_dict["R"]).reshape((3, 3))
        camera_info_dict["P"] = np.array(camera_info_dict["P"]).reshape((3, 4))
        return camera_info_dict


# AXIS_CONVERT_MAT_A2B = np.array(
#     [[0, 0, -1, 0],
#      [-1, 0, 0, 0],
#      [0, -1, 0, 0],
#      [0, 0, 0, 1]], dtype=np.float64
# )
AXIS_CONVERT_MAT_A2B = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], dtype=np.float64
)

if __name__ == "__main__":
    args = parse_args()

    path_to_pose_tsv = args.path_to_pose_tsv
    target_dir = os.path.dirname(path_to_pose_tsv)

    df_pose = pd.read_csv(path_to_pose_tsv, sep="\t", index_col=0)
    n = len(df_pose)
    pose_xyz = df_pose[['x', 'y', 'z']].values
    pose_quat = df_pose[['qx', 'qy', 'qz', 'qw']].values
    rotation_mat = Rotation.from_quat(pose_quat).as_matrix()
    mat = np.tile(np.eye(4), (n, 1, 1))
    mat[:, 0:3, 0:3] = rotation_mat
    mat[:, 0:3, 3:4] = pose_xyz.reshape((n, 3, 1))

    image_path_list = sorted(glob(f"{target_dir}/images/*.png"))

    # convert axis
    mat = AXIS_CONVERT_MAT_A2B.T @ mat @ AXIS_CONVERT_MAT_A2B

    output_dir = f"{target_dir}/sparse"
    os.makedirs(output_dir, exist_ok=True)
    camera_id = 1  # Fixed

    # make cameras.txt
    camera_info = load_camera_info_from_yaml(f"{target_dir}/camera_info.yaml")
    image = cv2.imread(image_path_list[0])
    h, w = image.shape[:2]
    f_cameras = open(f"{output_dir}/cameras.txt", "w")
    f_cameras.write(f"{camera_id} PINHOLE {w} {h} {camera_info['K'][0, 0]} {camera_info['K'][1, 1]} {camera_info['K'][0, 2]} {camera_info['K'][1, 2]}\n")

    # make images.txt
    quat = Rotation.from_matrix(mat[:, 0:3, 0:3]).as_quat()
    f_image = open(f"{output_dir}/images.txt", "w")
    for i in range(n):
        q = quat[i]
        f_image.write(f"{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {pose_xyz[i][0]} {pose_xyz[i][1]} {pose_xyz[i][2]} {camera_id} {os.path.basename(image_path_list[i])}\n\n")

    # make points3D.txt
    Path(f"{output_dir}/points3D.txt").touch()
