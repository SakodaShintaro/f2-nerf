""" A script to concatenate the test_images of the learned results with the true images
"""

import os
import argparse
from glob import glob
import cv2
import subprocess


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("test_images_dir", type=str)
    parse.add_argument("gt_images_dir", type=str)
    parse.add_argument("--prefix", type=str, default="color_20000_")
    args = parse.parse_args()
    return args


def put_text(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    args = parse_args()
    test_images_dir = args.test_images_dir
    gt_images_dir = args.gt_images_dir
    prefix = args.prefix
    test_image_path_list = sorted(glob(f"{test_images_dir}/{prefix}*.png"))
    if len(test_image_path_list) == 0:
        print(f"No test images found in {test_images_dir}")
        exit(1)
    save_dir = f"{test_images_dir}/../test_images_concat/"
    os.makedirs(save_dir, exist_ok=True)
    for i, test_image_path in enumerate(test_image_path_list):
        print(test_image_path)
        test_image_name = os.path.basename(test_image_path)
        frame_no = int(test_image_name.replace(".png", "").replace(prefix, ""))
        gt_image_path = f"{gt_images_dir}/{frame_no:08d}.png"
        test_image = cv2.imread(test_image_path)
        gt_image = cv2.imread(gt_image_path)
        h = min(test_image.shape[0], gt_image.shape[0], 384)
        w = min(test_image.shape[1], gt_image.shape[1], 768)
        test_image = cv2.resize(test_image, (w, h))
        gt_image = cv2.resize(gt_image, (w, h))
        put_text(test_image, f"NeRF result (frame={frame_no:04d})", 10, 30)
        put_text(gt_image, f"Ground Truth (frame={frame_no:04d})", 10, 30)
        concat_image = cv2.hconcat([test_image, gt_image])
        cv2.imwrite(f"{save_dir}/{i:08d}.png", concat_image)
    subprocess.run("ffmpeg -y -r 10 -f image2 -i %08d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p ../output.mp4",
                   shell=True, cwd=save_dir)
