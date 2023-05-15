#!/bin/bash

set -eux

cd $(dirname $0)

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

ros2 run my_image_proc undistort_node \
    --ros-args --remap src_image:=/sensing/camera/c1/image/compressed \
    --ros-args --remap src_info:=/sensing/camera/c1/camera_info \
    --ros-args --remap resized_image:=/sensing/camera/c1/image_rect_resized \
    --ros-args --remap resized_info:=/sensing/camera/c1/image_rect_resized_info
