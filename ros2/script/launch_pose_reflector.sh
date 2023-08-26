#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to pose_reflector --cmake-args -DCMAKE_BUILD_TYPE=Release

set +eux
source install/setup.bash
set -eux

ros2 run pose_reflector pose_reflector_node
