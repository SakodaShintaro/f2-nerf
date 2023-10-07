#!/bin/bash

set -eux

TRAIN_RESULT_DIR=$(readlink -f $1)

cd $(dirname $0)/../build/

make -j $(nproc)

./main walk ${TRAIN_RESULT_DIR}
