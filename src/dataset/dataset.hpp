//
// Created by ppwang on 2022/5/7.
//

#ifndef F2_NERF__DATASET_HPP_
#define F2_NERF__DATASET_HPP_

#include "../Common.h"

#include <torch/torch.h>

#include <string>
#include <tuple>
#include <vector>

struct alignas(32) Rays
{
  torch::Tensor origins;
  torch::Tensor dirs;
};

struct alignas(32) BoundedRays
{
  torch::Tensor origins;
  torch::Tensor dirs;
  torch::Tensor bounds;  // near, far
};

class Dataset
{
  using Tensor = torch::Tensor;

public:
  Dataset(const std::string & data_path, const std::string & output_dir);

  void NormalizeScene();
  void SaveInferenceParams() const;

  // Img2WorldRay
  static Rays Img2WorldRay(const Tensor & pose, const Tensor & intri, const Tensor & ij);

  // Rays
  BoundedRays RaysOfCamera(int idx, int reso_level = 1);
  std::tuple<BoundedRays, Tensor, Tensor> RandRaysData(int batch_size);

  // variables
  int n_images_ = 0;
  Tensor poses_, intri_, dist_params_, bounds_;
  Tensor center_;
  float radius_;
  int height_, width_;
  Tensor image_tensors_;

  // dirs
  const std::string output_dir_;
};

#endif  // F2_NERF__DATASET_HPP_
