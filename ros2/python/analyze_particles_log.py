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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Path to the log dir')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_dir = args.log_dir
    save_dir = f"{log_dir}/../particles_plot"
    os.makedirs(save_dir, exist_ok=True)
    log_file_list = sorted(glob(f"{log_dir}/*.tsv"))
    trajectory_x = list()
    trajectory_y = list()

    score_min = float("inf")
    score_max = -float("inf")
    for log_file in log_file_list:
        # plot current search result
        df = pd.read_csv(log_file, sep="\t")
        score = df["score"].values
        score_min = min(score_min, score.min())
        score_max = max(score_max, score.max())

    for log_file in tqdm(log_file_list):
        # plot the trajectory
        plt.plot(trajectory_x, trajectory_y, 'b')

        # plot current search result
        df = pd.read_csv(log_file, sep="\t")
        vec = df[["m03", "m13", "m23"]].values
        score = df["score"].values
        sc = plt.scatter(vec[:, 2], vec[:, 0], vmin=score_min, vmax=score_max, c=score, cmap=cm.seismic)
        plt.colorbar(sc)
        plt.axis('equal')
        plt.xlabel("z")
        plt.ylabel("x")
        plt.gca().invert_yaxis()
        save_path = f"{save_dir}/{log_file.split('/')[-1].split('.')[0]}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()

        best_index = score.argmin()
        trajectory_x.append(vec[best_index, 2])
        trajectory_y.append(vec[best_index, 0])

    subprocess.run("ffmpeg -y -r 10 -f image2 -i %08d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2: trunc(ih/2)*2\" ../output.mp4",
                   shell=True, cwd=save_dir)
