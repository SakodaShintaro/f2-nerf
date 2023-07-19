#!/bin/bash
set -eux

cd $(dirname $0)/../

cmake --build build --target main --config RelWithDebInfo -j8
python3 scripts/run.py --config-name=my_dataset
