#!/bin/bash

set -eux

cd $(dirname $0)/../
ROSBAG_PATH=$(readlink -f $1)

colcon build --symlink-install --packages-up-to pose_and_image_publisher --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
source install/setup.bash
set -u

rm -rf ./result_images/trial

ros2 service call /trigger_node_srv std_srvs/srv/SetBool "{data: true}"

# IMAGE_TOPIC_NAME=/sensing/camera/c1/image_rect_resized
IMAGE_TOPIC_NAME=/sensing/camera/traffic_light/image_raw
POSE_TOPIC_NAME=/localization/pose_twist_fusion_filter/biased_pose_with_covariance

ros2 bag play --rate 1 $ROSBAG_PATH \
              --topics ${IMAGE_TOPIC_NAME} \
                       ${POSE_TOPIC_NAME} \
                       /tf \
                       /tf_static \
              --remap ${IMAGE_TOPIC_NAME}:=image \
                      ${POSE_TOPIC_NAME}:=initial_pose_with_covariance

mkdir -p ./result_images/trial/log
mv ~/.ros/log/* ./result_images/trial/log/

python3 python/analyze_particles_log.py ./result_images/trial/particles/
python3 python/concat_test_images_result.py ./result_images/trial/
python3 python/analyze_ros2_log.py ./result_images/trial/log/nerf_based_localizer*
