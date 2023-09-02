""" A script to rectify images
"""
import argparse
from glob import glob
import cv2
from util_camera_info import load_camera_info_from_yaml
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_target_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_to_target_dir = args.path_to_target_dir
    path_to_image_dir = f"{path_to_target_dir}/images_original"
    path_to_camera_info_yaml = f"{path_to_target_dir}/camera_info.yaml"

    image_path_list = sorted(glob(f"{path_to_image_dir}/*.png"))
    camera_info = load_camera_info_from_yaml(path_to_camera_info_yaml)
    save_dir = f"{os.path.abspath(path_to_image_dir)}/../images"
    os.makedirs(save_dir, exist_ok=True)

    for image_path in tqdm(image_path_list):
        image = cv2.imread(image_path)
        image_rect = cv2.undistort(image, camera_info["K"], camera_info["D"])
        save_path = f"{save_dir}/{os.path.basename(image_path)}"
        cv2.imwrite(save_path, image_rect)
