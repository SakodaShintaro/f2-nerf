#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to ros2-f2-nerf --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

rm -rf ~/.ros/log/*

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/f2-nerf/External/libtorch/lib/
ros2 run ros2-f2-nerf nerf_based_localizer \
    --ros-args --params-file ./src/ros2-f2-nerf/config/parameters1.yaml \
    --ros-args --param save_image:=true \
    --ros-args --param save_particles:=true

# Debug
# ros2 run --prefix 'gdb -ex run --args' ros2-f2-nerf nerf_based_localizer
