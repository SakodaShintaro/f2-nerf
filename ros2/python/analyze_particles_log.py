""" Analyze the log file of the particle.
"""

import argparse
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from tqdm import tqdm
import subprocess
import numpy as np
import gtsam
from concurrent.futures import ProcessPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Path to the log dir')
    return parser.parse_args()


def plot_arrow(pose_mat, weight):
    orientation_mat = pose_mat[0:3, 0:3]
    position = pose_mat[0:3, 3]
    vec = orientation_mat @ np.array([0, 0, -0.5]) * weight
    color = (weight, 1 - weight, 0)
    plt.arrow(position[2], position[0],
              vec[2], vec[0], color=color, width=0.1 * weight)


def compute_rotation_average(rotations, weights):
    # Simple averaging does not use weighted average or k means.
    # https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf
    # section 5.3 Algorithm 1

    epsilon = 0.000001
    max_iters = 300
    R = rotations[0]
    for _ in range(max_iters):
        rot_sum = np.zeros((3))
        for i, rot in enumerate(rotations):
            rot_sum = rot_sum + weights[i] * gtsam.Rot3.Logmap(gtsam.Rot3(R.transpose() @ rot))

        if np.linalg.norm(rot_sum) < epsilon:
            return R
        else:
            r = gtsam.Rot3.Expmap(rot_sum).matrix()
            s = R @ r
            R = gtsam.Rot3(s).matrix()
    return R


def plot_function(
        log_file: str,
        trajectory_x: list,
        trajectory_y: list,
        weight_max: float,
        xlim: tuple,
        ylim: tuple,
        save_dir: str):
    # plot the trajectory
    plt.plot(trajectory_x, trajectory_y, 'b')

    # plot current search result
    df = pd.read_csv(log_file, sep="\t")

    rotations = df[["m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22"]].values.reshape(-1, 3, 3)
    positions = df[["m03", "m13", "m23"]].values
    weights = df["weight"].values

    curr_rotation = compute_rotation_average(rotations, weights)
    curr_position = np.average(positions, weights=weights, axis=0)
    weights /= weight_max

    for i, row in df.iterrows():
        pose = row.values[0:12].reshape(3, 4)
        weight = row.values[12]
        plot_arrow(pose, weight)
    # sc = plt.scatter(vec[:, 2], vec[:, 0], vmin=score_min, vmax=score_max, c=score, cmap=cm.seismic)
    # plt.colorbar(sc)
    plt.xlabel("z")
    plt.ylabel("x")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().invert_yaxis()
    save_path = f"{save_dir}/{log_file.split('/')[-1].split('.')[0]}.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    log_dir = args.log_dir
    save_dir = f"{log_dir}/../particles_plot"
    os.makedirs(save_dir, exist_ok=True)
    log_file_list = sorted(glob(f"{log_dir}/*.tsv"))
    trajectory_x = list()
    trajectory_y = list()

    weight_min = float("inf")
    weight_max = -float("inf")
    for log_file in log_file_list:
        # get weight range
        df = pd.read_csv(log_file, sep="\t")
        weights = df["weight"].values
        weight_min = min(weight_min, weights.min())
        weight_max = max(weight_max, weights.max())

        # get trajectory
        positions = df[["m03", "m13", "m23"]].values
        curr_position = np.average(positions, weights=weights, axis=0)
        trajectory_x.append(curr_position[2])
        trajectory_y.append(curr_position[0])

    # check plot range
    plt.plot(trajectory_x, trajectory_y, 'b')
    plt.axis('equal')
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.close()

    N = len(log_file_list)
    progress = tqdm(total=N)
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        for i in range(N):
            future = executor.submit(plot_function,
                                     log_file_list[i], trajectory_x[:i], trajectory_y[:i], weight_max, xlim, ylim, save_dir)
            future.add_done_callback(lambda _: progress.update())

    subprocess.run(
        "ffmpeg -y -r 10 -f image2 -i %08d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2: trunc(ih/2)*2\" ../output.mp4",
        shell=True,
        cwd=save_dir)
