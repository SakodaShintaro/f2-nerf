#!/bin/bash

set -eux

cd $(dirname $0)/../

# スクリプトが終了する際にすべてのバックグラウンドプロセスを停止するためのトラップを設定
trap "kill 0" EXIT

./script/check.sh ./src/ros2-f2-nerf/config/parameters_awsim.yaml &

./script/launch_pose_and_image_publisher.sh ~/data/converted/AWSIM/20230717_loop/
