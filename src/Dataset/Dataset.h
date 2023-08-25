//
// Created by ppwang on 2022/5/7.
//

#ifndef SANR_DATASET_H
#define SANR_DATASET_H

#pragma once
#include "../Common.h"
#include "../Utils/CameraUtils.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <string>
#include <tuple>
#include <vector>

#define DATA_TRAIN_SET 1
#define DATA_TEST_SET 2
#define DATA_VAL_SET 4

class Dataset
{
  using Tensor = torch::Tensor;

public:
  Dataset(const YAML::Node & config);

  void NormalizeScene();
  void SaveInferenceParams() const;

  const YAML::Node config_;

  // Img2WorldRay
  static Rays Img2WorldRay(const Tensor & pose, const Tensor & intri, const Tensor & ij);

  // Rays
  BoundedRays RaysOfCamera(int idx, int reso_level = 1);
  std::tuple<BoundedRays, Tensor, Tensor> RandRaysData(int batch_size, int sets);

  // variables
  int n_images_ = 0;
  Tensor poses_, intri_, dist_params_, bounds_;
  Tensor center_;
  float radius_;
  int height_, width_;
  std::vector<int> train_set_, test_set_, val_set_, split_info_;
  Tensor image_tensors_;

  // learnable pose
  static Tensor vec2skew(Tensor v); // v(n, 3) -> return (n, 3, 3)
  static Tensor so3_to_SO3(Tensor r); // r(n, r) -> return (n, 3, 3)
  static Tensor convert_r_and_t_to_3x4(Tensor r, Tensor t); // r(n, 3), t(n, 3) -> return (n, 3, 4)
  Tensor learnable_pos_;
  Tensor learnable_ori_;
};

#endif  // SANR_DATASET_H