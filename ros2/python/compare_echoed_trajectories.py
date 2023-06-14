""" A script to plot the echoed trajectories
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_list", type=str, nargs="+")
    parser.add_argument("--label_list", type=str, nargs="+")
    return parser.parse_args()


def parse_ros_output(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # '---'で区切られた各ブロックを見つける
    blocks = content.split('---')[:-1]
    df = pd.DataFrame(columns=["x", "y", "z", "qx", "qy", "qz", "qw"])

    for block in tqdm(blocks):
        # 余分な空白行を取り除く
        block = block.strip()
        if not block:
            continue
        # yamlライブラリを使って各ブロックを解析する
        block_data = yaml.safe_load(block)
        df.loc[len(df)] = [
            block_data["pose"]["pose"]["position"]["x"],
            block_data["pose"]["pose"]["position"]["y"],
            block_data["pose"]["pose"]["position"]["z"],
            block_data["pose"]["pose"]["orientation"]["x"],
            block_data["pose"]["pose"]["orientation"]["y"],
            block_data["pose"]["pose"]["orientation"]["z"],
            block_data["pose"]["pose"]["orientation"]["w"]
        ]

    return df


if __name__ == "__main__":
    args = parse_args()
    log_list = args.log_list
    label_list = args.label_list
    assert len(log_list) == len(label_list)

    for log_file, label in zip(log_list, label_list):
        df = parse_ros_output(log_file)
        plt.plot(df["x"], df["y"], label=label)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    save_path = "echoed_trajectory.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved to {save_path}")
