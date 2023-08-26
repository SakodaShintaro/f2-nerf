#!/bin/bash

set -eux

cd $(dirname $0)

./build_and_exec_training.sh

./build_and_exec_inference.sh ../exp/20230717_loop/runtime_config.yaml

../ros2/script/regression_test.sh
