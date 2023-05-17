#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build

set +u
source install/setup.bash
set -u

ros2 run pose_and_image_publisher pose_and_image_publisher ~/data/converted/20230501_try1/
