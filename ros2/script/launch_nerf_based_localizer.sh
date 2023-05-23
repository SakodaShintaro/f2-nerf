#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to ros2-f2-nerf --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/f2-nerf/External/libtorch/lib/
ros2 run ros2-f2-nerf nerf_based_localizer \
    --ros-args --params-file ./src/ros2-f2-nerf/config/parameters.yaml \
    --ros-args --remap image:=/sensing/camera/c1/image_rect_resized \
    --ros-args --remap initial_pose_with_covariance:=/localization/pose_twist_fusion_filter/biased_pose_with_covariance \
    --ros-args --remap nerf_service:=/localization/pose_estimator/ndt_align_srv \
    --ros-args --remap trigger_node_srv:=/localization/pose_estimator/trigger_node \
    --ros-args --remap nerf_pose_with_covariance:=/localization/pose_estimator/pose_with_covariance
