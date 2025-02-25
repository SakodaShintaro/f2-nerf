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
DATASET_PATH=$(readlink -f $1)

# move to the directory where this script is
cd $(dirname $0)

# set for command line
export QT_QPA_PLATFORM=offscreen

# Recommended CAMERA values: OPENCV for perspective, OPENCV_FISHEYE for fisheye.
CAMERA=OPENCV

USE_GPU=1

# Run COLMAP.

colmap feature_extractor \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.use_gpu "$USE_GPU"

colmap exhaustive_matcher \
    --database_path "$DATASET_PATH"/database.db \
    --SiftMatching.use_gpu "$USE_GPU"

mkdir -p "$DATASET_PATH"/sparse

colmap mapper \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --output_path "$DATASET_PATH"/sparse \
    --Mapper.ba_global_pba_gpu_index 0

python3 convert_pose_tsv_to_colmap_format.py "$DATASET_PATH"/pose.tsv
mkdir -p "$DATASET_PATH"/pose_aligned

colmap model_aligner \
  --input_path "$DATASET_PATH"/sparse/0/ \
  --output_path "$DATASET_PATH"/pose_aligned \
  --ref_images_path "$DATASET_PATH"/reference_trajectory.txt \
  --robust_alignment_max_error 1 \
  --ref_is_gps 0

python3 colmap2poses.py --data_dir $DATASET_PATH

i=$SECONDS
((sec=i%60, min=(i%3600)/60, hrs=i/3600))
timestamp=$(printf "%d:%02d:%02d" "$hrs" "$min" "$sec")
echo "Processing time is $timestamp"
