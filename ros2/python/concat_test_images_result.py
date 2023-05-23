""" A script to concatenate the test_images of the learned results with the true images
"""

import os
import argparse
from glob import glob
import cv2
import subprocess
from tqdm import tqdm


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("target_dir", type=str)
    args = parse.parse_args()
    return args


def put_text(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.target_dir
    pred_image_path_list = sorted(glob(f"{target_dir}/pred/*.png"))
    if len(pred_image_path_list) == 0:
        print(f"No test images found in {target_dir}")
        exit(1)
    save_dir = f"{target_dir}/pred_images_concat/"
    os.makedirs(save_dir, exist_ok=True)
    for i, pred_image_path in enumerate(tqdm(pred_image_path_list)):
        test_image_name = os.path.basename(pred_image_path)
        frame_no = int(test_image_name.replace(".png", ""))
        gt_image_path = f"{target_dir}/gt/{frame_no:08d}.png"
        plot_image_path = f"{target_dir}/particles_plot/{frame_no:08d}.png"
        pred_image = cv2.imread(pred_image_path)
        gt_image = cv2.imread(gt_image_path)
        plot_image = cv2.imread(plot_image_path)
        h = 420
        w = 540
        h2w_scale = pred_image.shape[1] / pred_image.shape[0]
        shape = (int(h // 2 * h2w_scale), h // 2)
        pred_image = cv2.resize(pred_image, shape)
        gt_image = cv2.resize(gt_image, shape)
        plot_image = cv2.resize(plot_image, (w, h))
        put_text(pred_image, f"NeRF", 10, 30)
        put_text(gt_image, f"GT", 10, 30)
        concat_image = cv2.vconcat([pred_image, gt_image])
        concat_image = cv2.hconcat([concat_image, plot_image])
        cv2.imwrite(f"{save_dir}/{i:08d}.png", concat_image)
    subprocess.run("ffmpeg -y -r 10 -f image2 -i %08d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2: trunc(ih/2)*2\" ../output_concat.mp4",
                   shell=True, cwd=save_dir)
