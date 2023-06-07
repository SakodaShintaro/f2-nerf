import argparse
import pandas as pd
import os
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_pose_tsv", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path_to_pose_tsv = args.path_to_pose_tsv
    target_dir = os.path.dirname(path_to_pose_tsv)

    df_pose = pd.read_csv(path_to_pose_tsv, sep="\t", index_col=0)
    n = len(df_pose)
    xyz = df_pose[['x', 'y', 'z']].values

    image_path_list = sorted(glob(f"{target_dir}/images/*.png"))

    # make images.txt
    f_image = open(f"{target_dir}/reference_trajectory.txt", "w")
    for i in range(n):
        f_image.write(
            f"{os.path.basename(image_path_list[i])} {xyz[i][0]} {xyz[i][1]} {xyz[i][2]}\n")
