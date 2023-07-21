""" A script to make movie by plot score and concat images
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


def put_text(image, text, x, y, scale, color, outline_color=(0,0,0)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, outline_color, 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
    return image


if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir

    image_gt = cv2.imread(f"{result_dir}/image_01_gt.png")
    image_before = cv2.imread(f"{result_dir}/image_02_before.png")
    image_noised = cv2.imread(f"{result_dir}/image_03_noised.png")
    image_after_path_list = sorted(glob(f"{result_dir}/image_04_after_*.png"))
    image_after_list = [cv2.imread(path) for path in image_after_path_list]

    x = 5
    y = 15
    scale = 0.4
    color = (0, 0, 255)

    put_text(image_gt, "(1) GT Image", x, y, scale, color)
    put_text(image_before, "(2) NeRF @ GT Pose", x, y, scale, color)
    put_text(image_noised, "(3) NeRF @ Noised Pose", x, y, scale, color)

    df = pd.read_csv(f"{result_dir}/../score.tsv", sep="\t")
    save_path = f"{result_dir}/../score_plot.png"

    for i, image_after in enumerate(tqdm(image_after_list)):
        # plot
        plt.figure(figsize=(8, 1.5))
        plt.plot(df["iteration"], df["score"])
        plt.axvline(x=i, color='r', linestyle='--')
        plt.xlabel("iteration")
        plt.ylabel("score")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        plot_image = cv2.imread(save_path)

        # concat
        put_text(image_after, f"(4) NeRF @ Optimized Pose {i:02d}", x, y, scale, color)

        image_con1 = cv2.hconcat([image_gt, image_before])
        image_con2 = cv2.hconcat([image_noised, image_after])
        width = image_con1.shape[1]
        height = int(plot_image.shape[0] *
                     width / plot_image.shape[1]) // 2 * 2
        plot_image = cv2.resize(plot_image, (width, height))
        image = cv2.vconcat([image_con1, image_con2, plot_image])
        cv2.imwrite(f"{result_dir}/image_05_concat_{i:04d}.png", image)

    subprocess.run("ffmpeg -y -r 5 -f image2 -i image_05_concat_%04d.png -vcodec libx264 -pix_fmt yuv420p ../concat_movie.mp4",
                   shell=True, cwd=result_dir)
