//
// Created by ppwang on 2022/5/7.
//

#ifndef F2_NERF__DATASET_HPP_
#define F2_NERF__DATASET_HPP_

#include "common.hpp"
#include "rays.hpp"

#include <torch/torch.h>

#include <string>
#include <tuple>
#include <vector>

class Dataset
{
  using Tensor = torch::Tensor;

public:
  Dataset(const std::string & data_path);

  void save_inference_params(const std::string & output_dir) const;

  BoundedRays get_all_rays_of_camera(int idx);

  std::tuple<BoundedRays, Tensor, Tensor> sample_random_rays(int batch_size);

  int n_images_ = 0;
  Tensor poses_, intrinsics_, dist_params_, bounds_;
  Tensor center_;
  float radius_;
  int height_, width_;
  Tensor image_tensors_;
};

#endif  // F2_NERF__DATASET_HPP_
