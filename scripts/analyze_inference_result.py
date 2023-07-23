""" A script to analyze inference result
"""

import argparse
import cv2
from glob import glob
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir

    frame_dir_list = sorted(glob(f"{result_dir}/*/"))
    for frame_dir in tqdm(frame_dir_list):
        df = pd.read_csv(f"{frame_dir}/position.tsv", sep="\t")
        # get the overall min and max scores
        min_score = df['score'].min()
        max_score = df['score'].max()

        original = df[df['name'] == 'original']
        plt.scatter(original['x'], original['z'], c=original['score'],
                    cmap='viridis', vmin=min_score, vmax=max_score, label='Original')

        for i in range(8):
            noised = df[df['name'] == f'noised_{i}']
            optimized = df[df['name'] == f'optimized_{i}']

            plt.scatter(noised['x'], noised['z'], c=noised['score'],
                        cmap='viridis', vmin=min_score, vmax=max_score, label=f'Noised_{i}')
            plt.scatter(optimized['x'], optimized['z'],
                        c=optimized['score'], cmap='viridis', vmin=min_score, vmax=max_score, label=f'Optimized_{i}')

            # Draw an arrow from the noised position to the optimized position
            plt.annotate('', xy=(optimized['x'].values[0], optimized['z'].values[0]),
                         xytext=(noised['x'].values[0], noised['z'].values[0]),
                         arrowprops=dict(facecolor='gray', shrink=0.05))

        plt.xlabel('x')
        plt.ylabel('z')
        plt.colorbar(label='Score')
        plt.savefig(f"{frame_dir}/plot_result.png",
                    bbox_inches="tight", pad_inches=0.05)
        plt.close()
