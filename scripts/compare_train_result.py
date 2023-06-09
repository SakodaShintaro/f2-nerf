""" A script to compare the training results.
"""

import argparse
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_list", type=str, nargs="+")
    parser.add_argument("--label_list", type=str, nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_list = args.log_list
    label_list = args.label_list
    assert len(log_list) == len(label_list)

    for j, log in enumerate(log_list):
        f = open(log, "r")
        lines = f.readlines()
        f.close()
        iteration_list = []
        psnr_list = []
        for line in lines:
            line = line.strip()
            elements = line.split(":")
            curr_dict = {}
            for i in range(0, len(elements) - 1):
                elements[i] = elements[i].strip()
                elements[i + 1] = elements[i + 1].strip()
                name = elements[i].split(" ")[-1]
                value = float(elements[i + 1].split(" ")[0])
                curr_dict[name] = value
            iteration_list.append(curr_dict["Iter"])
            psnr_list.append(curr_dict["PSNR"])
        plt.plot(iteration_list, psnr_list, label=label_list[j])
    plt.xlabel("Step")
    plt.ylabel("PSNR")
    plt.legend()
    save_dir = os.path.dirname(log_list[0])
    plt.savefig(f"{save_dir}/train_result.png", bbox_inches="tight", pad_inches=0.05)
    print(f"Save the training result to {save_dir}/train_result.png")
