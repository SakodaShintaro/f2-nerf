#!/bin/bash

set -eux

cd $(dirname $0)

./build_and_exec_training.sh

./build_and_exec_test.sh ../exp/20230609_base_link_logiee/runtime_config.yaml
