""" A script to analyze ros2 log of nerf_based_localizer node.
"""

import argparse
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_file = args.log_file
    f = open(log_file, 'r')
    lines = f.readlines()
    f.close()
    score_list = list()
    for line in lines:
        if "score = " not in line:
            continue
        line = line.strip()
        content = line.split("[nerf_based_localizer]")[-1][2:]
        score = float(content.replace("score = ", ""))
        score_list.append(score)
    plt.plot(score_list)
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.ylim(bottom=0.0)
    save_path = f"{os.path.dirname(log_file)}/score.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved to {save_path}")
