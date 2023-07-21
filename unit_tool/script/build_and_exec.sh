#!/bin/bash

set -eux

cd $(dirname $0)/../build/

make -j $(nproc)

rm -rf result_images movie.mp4

./f2-nerf_unit_tool

python3 ../python/make_movie.py ./result_images/
