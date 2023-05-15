#!/bin/bash

set -eux

cd $(dirname $0)

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/f2-nerf/External/libtorch/lib/
ros2 run ros2-f2-nerf nerf_based_localizer \
    --ros-args --remap image:=/sensing/camera/c1/image_rect_resized \
    --ros-args --remap initial_pose_with_covariance:=/localization/pose_twist_fusion_filter/biased_pose_with_covariance
