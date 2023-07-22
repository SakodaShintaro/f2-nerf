#!/bin/bash

set -eux

CONFIG_PATH=$(readlink -f $1)

cd $(dirname $0)/../build/

make -j $(nproc)

rm -rf result_images movie.mp4

./f2-nerf_unit_tool ${CONFIG_PATH}

python3 ../python/make_movie.py ./result_images/
