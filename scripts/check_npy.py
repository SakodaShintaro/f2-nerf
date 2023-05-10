""" A script to check the npy file.
"""

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_npy_file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    npy = np.load(args.path_to_npy_file)
    print(npy.shape)
    np.set_printoptions(precision=6, suppress=True)
    pose = npy[0, 0:12].reshape((3, 4))
    print(pose)
