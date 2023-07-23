#!/bin/bash

set -eux

CONFIG_PATH=$(readlink -f $1)

cd $(dirname $0)/../build/

make -j $(nproc)

rm -rf result_images movie.mp4

./inference_tool ${CONFIG_PATH}

python3 ../scripts/analyze_inference_result.py ./inference_result/
