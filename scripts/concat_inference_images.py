import cv2
import os
import argparse
from glob import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1', type=str)
    parser.add_argument('dir2', type=str)
    parser.add_argument('train_result_dir', type=str)
    return parser.parse_args()


def pad_images(image1, image2):
    # 画像の高さが異なる場合、高さを揃える
    if image1.shape[0] != image2.shape[0]:
        # 差を計算
        diff = abs(image1.shape[0] - image2.shape[0])
        # 短い方の画像にパディングを追加
        if image1.shape[0] < image2.shape[0]:
            padding = np.zeros(
                (diff, image1.shape[1], image1.shape[2]), dtype=image1.dtype)
            image1 = np.vstack((image1, padding))
        else:
            padding = np.zeros(
                (diff, image2.shape[1], image2.shape[2]), dtype=image2.dtype)
            image2 = np.vstack((image2, padding))

    return image1, image2


def main():
    args = parse_args()
    dir1 = args.dir1
    dir2 = args.dir2
    train_result_dir = args.train_result_dir

    os.makedirs(train_result_dir, exist_ok=True)

    frame_dirs = sorted(glob(f"{dir1}/*/"))
    files2 = sorted(glob(f"{dir2}/*.png"))

    for frame_dir, file2 in zip(frame_dirs, files2):
        image1 = cv2.imread(f"{frame_dir}/plot_result_last.png")
        image2 = cv2.imread(file2)

        # image1を2倍に拡大
        image1 = cv2.resize(image1, None, fx=2, fy=2)

        # 画像の高さを揃える
        image1, image2 = pad_images(image1, image2)

        # 連結
        new_image = cv2.hconcat([image1, image2])

        # 保存
        output_path = f"{train_result_dir}/{os.path.basename(file2)}"
        cv2.imwrite(output_path, new_image)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
