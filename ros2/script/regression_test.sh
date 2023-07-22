#!/bin/bash

set -eux

cd $(dirname $0)/../

# スクリプトが終了する際にすべてのバックグラウンドプロセスを停止するためのトラップを設定
trap "kill 0" EXIT

./script/check.sh \
    ./src/ros2-f2-nerf/config/parameters_awsim.yaml\
    /home/sakoda/work/f2-nerf/exp/20230717_loop/test/record/runtime_config.yaml &

./script/launch_pose_and_image_publisher.sh ~/data/converted/AWSIM/20230717_loop/

# 秒数表示
echo "${SECONDS} sec"
