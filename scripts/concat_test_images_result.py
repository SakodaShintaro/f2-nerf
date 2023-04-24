""" A script to concatenate the test_images of the learned results with the true images
"""

import os
import argparse
from glob import glob
import cv2


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("test_images_dir", type=str)
    parse.add_argument("gt_images_dir", type=str)
    args = parse.parse_args()
    return args


def put_text(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    args = parse_args()
    test_images_dir = args.test_images_dir
    gt_images_dir = args.gt_images_dir
    test_image_path_list = sorted(glob(f"{test_images_dir}/color*.png"))
    save_dir = f"{test_images_dir}/../test_images_concat/"
    os.makedirs(save_dir, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{test_images_dir}/../test_images_concat.mp4", codec,
                            10.0, (1280, 360))
    for test_image_path in test_image_path_list:
        print(test_image_path)
        test_image_name = os.path.basename(test_image_path)
        frame_no = int(test_image_name.replace(".png", "").split("_")[2])
        gt_image_path = f"{gt_images_dir}/{frame_no:08d}.png"
        test_image = cv2.imread(test_image_path)
        gt_image = cv2.imread(gt_image_path)
        h = min(test_image.shape[0], gt_image.shape[0])
        w = min(test_image.shape[1], gt_image.shape[1])
        test_image = cv2.resize(test_image, (w, h))
        gt_image = cv2.resize(gt_image, (w, h))
        put_text(test_image, f"NeRF result (frame={frame_no:04d})", 10, 30)
        put_text(gt_image, f"Ground Truth (frame={frame_no:04d})", 10, 30)
        concat_image = cv2.hconcat([test_image, gt_image])
        video.write(concat_image)
        cv2.imwrite(f"{save_dir}/{test_image_name}", concat_image)
    video.release()
