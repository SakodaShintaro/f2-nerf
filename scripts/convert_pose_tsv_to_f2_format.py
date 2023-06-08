import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os
import yaml


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


AXIS_CONVERT_MAT_A2B = np.array(
    [[0, 0, -1, 0],
     [-1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]], dtype=np.float64
)
AXIS_CONVERT_MAT_W2N = np.array(
    [[ 0, -1,  0,  0],
     [ 0,  0, +1,  0],
     [-1,  0,  0,  0],
     [ 0,  0,  0, +1]], dtype=np.float64
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

    # convert axis
    mat = AXIS_CONVERT_MAT_A2B.T @ mat @ AXIS_CONVERT_MAT_A2B
    mat = mat[:, 0:3, :]
    mat = mat.reshape((n, 12))

    # save camera meta
    camera_info = load_camera_info_from_yaml(f"{target_dir}/camera_info.yaml")
    k = camera_info["K"]
    camera_param = np.tile(k, (n, 1, 1))
    camera_param = camera_param.reshape((n, 9))

    dist_param = camera_info["D"][0:4]
    dist_param = np.tile(dist_param, (n, 1))

    bounds = np.array([[1.0, 30.0] for _ in range(n)])

    data = np.concatenate([mat, camera_param, dist_param, bounds], axis=1)
    np.save(os.path.join(target_dir, "cams_meta.npy"), data)
