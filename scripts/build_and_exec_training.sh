#!/bin/bash
set -eux

ROOT_DIR=$(readlink -f $(dirname $0)/../)
DATASET_PATH=$(readlink -f $1)

cd ${ROOT_DIR}
cmake --build build --config RelWithDebInfo -j8

cd ${ROOT_DIR}/ros2
colcon build --symlink-install --packages-up-to ros2-f2-nerf --cmake-args -DCMAKE_BUILD_TYPE=Release

cd ${ROOT_DIR}
python3 scripts/run.py --config-name=my_dataset dataset_path=${DATASET_PATH}
