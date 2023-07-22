#!/bin/bash

set -eux

PARAMETER_FILE=$(readlink -f $1)
RUNTIME_CONFIG=$(readlink -f $2)

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to ros2-f2-nerf --cmake-args -DCMAKE_BUILD_TYPE=Release

set +eux
source install/setup.bash
set -eux

rm -rf ~/.ros/log/*

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/f2-nerf/External/libtorch/lib/
ros2 run ros2-f2-nerf nerf_based_localizer \
    --ros-args --params-file $PARAMETER_FILE \
    --ros-args --param runtime_config_path:=$RUNTIME_CONFIG \
    --ros-args --param save_image:=true \
    --ros-args --param save_particles:=true

# Debug
# ros2 run --prefix 'gdb -ex run --args' ros2-f2-nerf nerf_based_localizer
