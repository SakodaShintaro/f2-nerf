import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os

parser = argparse.ArgumentParser()
parser.add_argument("path_to_pose_tsv", type=str)
args = parser.parse_args()

path_to_pose_tsv = args.path_to_pose_tsv
target_dir = os.path.dirname(path_to_pose_tsv)

df_pose = pd.read_csv(path_to_pose_tsv, sep="\t", index_col=0)
n = len(df_pose)
pose_xyz = df_pose[['x', 'y', 'z']].values
pose_xyz -= pose_xyz[0]
pose_quat = df_pose[['qx', 'qy', 'qz', 'qw']].values
rotation_mat = Rotation.from_quat(pose_quat).as_matrix()
mat = np.zeros((n, 3, 4))
mat[:, 0:3, 0:3] = rotation_mat
mat[:, 0:3, 3] = pose_xyz

# save pose
np.save(os.path.join(target_dir, "poses_render.npy"), mat)

# save camera meta
file_camera_info = open(f"{target_dir}/camera_info.txt")
line = file_camera_info.readline()
fx, cx, fy, cy = list(map(float, line.split(' ')))
camera_param = np.zeros((n, 3, 3))
camera_param[:, 0, 0] = fx
camera_param[:, 0, 2] = cx
camera_param[:, 1, 1] = fy
camera_param[:, 1, 2] = cy
camera_param[:, 2, 2] = 1
camera_param = camera_param.reshape((n, 9))

dist_param = np.zeros((n, 4))

bounds = np.array([[1.0, 30.0] for _ in range(n)])

mat = mat.reshape((n, 12))

data = np.concatenate([mat, camera_param, dist_param, bounds], axis=1)
np.save(os.path.join(target_dir, "cams_meta.npy"), data)
