#!/bin/bash

set -eux

DATASET_PATH=$(readlink -f $1)
cd $(dirname $0)

./build_and_exec_training.sh ${DATASET_PATH}

./build_and_exec_test.sh ../exp/$(basename ${DATASET_PATH})/runtime_config.yaml
