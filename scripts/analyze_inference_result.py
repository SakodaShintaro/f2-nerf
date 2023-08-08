""" A script to analyze inference result
"""

import argparse
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir

    ITERATION = 10

    frame_dir_list = sorted(glob(f"{result_dir}/*/"))
    for frame_dir in tqdm(frame_dir_list):
        df = pd.read_csv(f"{frame_dir}/position.tsv", sep="\t")
        # get the overall min and max scores
        min_score = df['score'].min()
        max_score = df['score'].max()

        original = df[df['name'] == 'original']
        plt.scatter(original['x'], original['y'], c=original['score'],
                    cmap='viridis', vmin=min_score, vmax=max_score, label='Original')

        for i in range(8):
            noised = df[df['name'] == f'noised_{i}']
            plt.scatter(noised['x'], noised['y'], s=50,
                        color="red", label=f'Noised_{i}')
            plt.scatter(noised['x'], noised['y'], c=noised['score'], s=25,
                        cmap='viridis', vmin=min_score, vmax=max_score, label=f'Noised_{i}')
            for j in range(ITERATION):
                optimized = df[df['name'] == f'optimized_{i}_{j:02d}']

                plt.scatter(optimized['x'], optimized['y'],
                            c=optimized['score'], cmap='viridis', vmin=min_score, vmax=max_score, label=f'Optimized_{i}_{j:02d}')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Score')
        save_path = os.path.abspath(f"{frame_dir}/plot_result.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Saved to {save_path}")

        # print only last pos
        original = df[df['name'] == 'original']
        plt.scatter(original['x'], original['y'], c=original['score'],
                    cmap='viridis', vmin=min_score, vmax=max_score, label='Original')

        for i in range(8):
            noised = df[df['name'] == f'noised_{i}']
            nx = noised['x'].values[0]
            ny = noised['y'].values[0]
            plt.scatter(nx, ny, s=50, color="red", label=f'Noised_{i}')
            plt.scatter(nx, ny, c=noised['score'], s=25,
                        cmap='viridis', vmin=min_score, vmax=max_score, label=f'Noised_{i}')
            optimized = df[df['name'] == f'optimized_{i}_{ITERATION - 1:02d}']
            ox = optimized['x'].values[0]
            oy = optimized['y'].values[0]
            plt.scatter(ox, oy,
                        c=optimized['score'], cmap='viridis', vmin=min_score, vmax=max_score, label=f'Optimized_{i}_{j:02d}')

            # plot arrow noised -> optimized
            plt.arrow(nx, ny, ox - nx, oy - ny, length_includes_head=True,
                      head_width=0.05, head_length=0.05, fc='k', ec='k')


        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Score')
        save_path = os.path.abspath(f"{frame_dir}/plot_result_last.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Saved to {save_path}")
