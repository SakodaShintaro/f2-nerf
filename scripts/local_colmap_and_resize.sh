#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eux

# Path to a directory `base/` with images in `base/images/`.
DATASET_PATH=$1

# set for command line
export QT_QPA_PLATFORM=offscreen

# Recommended CAMERA values: OPENCV for perspective, OPENCV_FISHEYE for fisheye.
CAMERA=OPENCV

USE_GPU=0

# Run COLMAP.

colmap feature_extractor \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.use_gpu "$USE_GPU"


# colmap exhaustive_matcher \
#     --database_path "$DATASET_PATH"/database.db \
#     --SiftMatching.use_gpu "$USE_GPU"

## Use if your scene has > 500 images
## Replace this path with your own local copy of the file.
## Download from: https://demuc.de/colmap/#download
# colmap vocab_tree_matcher \
#     --database_path "$DATASET_PATH"/database.db \
#     --VocabTreeMatching.vocab_tree_path /root/f2-nerf/vocab_tree_flickr100K_words32K.bin \
#     --SiftMatching.use_gpu "$USE_GPU"

colmap sequential_matcher \
    --database_path "$DATASET_PATH"/database.db \
    --SiftMatching.use_gpu "$USE_GPU"

mkdir -p "$DATASET_PATH"/sparse

colmap mapper \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --output_path "$DATASET_PATH"/sparse

mkdir -p "$DATASET_PATH"/dense

colmap image_undistorter \
    --image_path "$DATASET_PATH"/images \
    --input_path "$DATASET_PATH"/sparse/0 \
    --output_path "$DATASET_PATH"/dense \
    --output_type COLMAP
