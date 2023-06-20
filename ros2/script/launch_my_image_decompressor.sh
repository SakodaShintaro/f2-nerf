#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to my_image_decompressor --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

ros2 run my_image_decompressor my_image_decompressor \
    --ros-args --remap src_topic:=/sensing/camera/camera0/image_rect_color/compressed \
    --ros-args --remap dst_topic:=/sensing/camera/camera0/image_rect_color/decompressed
