#!/bin/bash

set -eux

CONFIG_PATH=$(readlink -f $1)

cd $(dirname $0)/../build/

make -j $(nproc)

./main walk ${CONFIG_PATH}
