#!/bin/bash

set -eux

TRAIN_RESULT_DIR=$(readlink -f $1)
DATASET_DIR=$(readlink -f $2)

cd $(dirname $0)/../build/

make -j $(nproc)

rm -rf inference_result result_images movie.mp4

./main infer ${TRAIN_RESULT_DIR} ${DATASET_DIR}

python3 ../scripts/analyze_inference_result.py ./inference_result/
