#!/usr/bin/env python3

""" A script to crop images.
"""

import argparse
import os
from glob import glob
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.target_dir

    input_dir = f"{target_dir}/images_original"
    output_dir = f"{target_dir}/images"

    os.makedirs(output_dir, exist_ok=True)
    image_list = sorted(glob(f"{input_dir}/*.png"))
    for image_path in tqdm(image_list):
        image = cv2.imread(image_path)
        cropped_image = image[0:850]
        save_path = f"{output_dir}/{os.path.basename(image_path)}"
        cv2.imwrite(save_path, cropped_image)
