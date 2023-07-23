#!/bin/bash

set -eux

cd $(dirname $0)

./build_and_exec_training.sh

../ros2/script/regression_test.sh
