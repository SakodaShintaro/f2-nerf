#!/bin/bash

set -eux

DATA_PATH=$(readlink -f $1)

cd $(dirname $0)/../

colcon build --symlink-install --packages-up-to pose_and_image_publisher --cmake-args -DCMAKE_BUILD_TYPE=Release

set +eux
source install/setup.bash
set -eux

rm -rf ./result_images/trial

ros2 service call /trigger_node_srv std_srvs/srv/SetBool "{data: true}"

ros2 run pose_and_image_publisher pose_and_image_publisher ${DATA_PATH}

mkdir -p ./result_images/trial/log
mv ~/.ros/log/* ./result_images/trial/log/

python3 python/analyze_particles_log.py ./result_images/trial/particles/
python3 python/concat_test_images_result.py ./result_images/trial/
python3 python/analyze_ros2_log.py ./result_images/trial/log/nerf_based_localizer*
