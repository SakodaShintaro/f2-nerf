#!/bin/bash

set -eux

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/f2-nerf/External/libtorch/lib/
ros2 run ros2-f2-nerf nerf_based_localizer

# Debug
# ros2 run --prefix 'gdb -ex run --args' ros2-f2-nerf nerf_based_localizer
