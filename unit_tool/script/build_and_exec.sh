#!/bin/bash

set -eux

cd $(dirname $0)/../build/

make -j $(nproc)

rm -f *.png movie.mp4

./f2-nerf_unit_tool

ffmpeg -r 10 \
       -i image_04_after_%04d.png \
       -vcodec libx264 \
       -pix_fmt yuv420p \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       -r 10 \
       movie.mp4
