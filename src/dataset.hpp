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

struct Dataset
{
  using Tensor = torch::Tensor;

public:
  Dataset(const std::string & data_path);

  void save_inference_params(const std::string & train_result_dir) const;

  Rays get_all_rays_of_camera(int idx);

  std::tuple<Rays, Tensor, Tensor> sample_random_rays(int batch_size);

  static std::vector<std::string> glob_image_paths(const std::string & input_dir);

  int n_images;
  Tensor poses, images, intrinsics, dist_params;
  Tensor center;
  float radius;
  int height, width;
};

#endif  // F2_NERF__DATASET_HPP_
