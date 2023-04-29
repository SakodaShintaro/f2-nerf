""" A script to add spli.npy to the data directory.
"""

import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_dir', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_to_dir = args.path_to_dir
    path_to_pose_tsv = f"{path_to_dir}/pose.tsv"
    df = pd.read_csv(path_to_pose_tsv, sep="\t", index_col=0)
    n = len(df)

    # train:0bit, test:1bit, val:2bit
    data = np.array([0b011 for _ in range(n)], dtype=np.uint8)
    data = data.reshape((n, 1))
    path_to_split = f"{path_to_dir}/split.npy"
    np.save(f"{path_to_dir}/split.npy", data)
