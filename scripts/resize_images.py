""" A script to resize images in a specified directory and below
"""

import argparse
import os
import cv2
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_dir", type=str)
    parser.add_argument("path_to_output_dir", type=str)
    parser.add_argument("--resize_factor", type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir = args.path_to_input_dir
    path_to_output_dir = args.path_to_output_dir
    resize_factor = args.resize_factor

    os.makedirs(path_to_output_dir, exist_ok=True)
    image_path_list = sorted(glob(f"{path_to_input_dir}/*.png"))
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        image = cv2.resize(
            image, (image.shape[1] // resize_factor, image.shape[0] // resize_factor))
        image_name = os.path.basename(image_path)
        cv2.imwrite(f"{path_to_output_dir}/{image_name}", image)
