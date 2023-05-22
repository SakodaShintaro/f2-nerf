#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to pose_and_image_publisher --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

rm -rf ./result_images/trial

ros2 service call /trigger_node_srv std_srvs/srv/SetBool "{data: true}"

ros2 run pose_and_image_publisher pose_and_image_publisher ~/data/converted/20230501_try1/

mkdir -p ./result_images/trial/log
mv ~/.ros/log/* ./result_images/trial/log/

python3 python/analyze_particles_log.py ./result_images/trial/particles/
