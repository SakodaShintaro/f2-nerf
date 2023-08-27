//
// Created by ppwang on 2022/6/20.
//

#pragma once
#include "../Common.h"
#include "Eigen/Eigen"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <memory>

constexpr int MAX_SAMPLE_PER_RAY = 1024;

struct SampleResultFlex
{
  using Tensor = torch::Tensor;
  Tensor pts;             // [ n_all_pts, 3 ]
  Tensor dirs;            // [ n_all_pts, 3 ]
  Tensor dt;              // [ n_all_pts, 1 ]
  Tensor t;               // [ n_all_pts, 1 ]
  Tensor pts_idx_bounds;  // [ n_rays, 2 ] // start, end
};

enum RunningMode { TRAIN, VALIDATE };

class PtsSampler
{
  using Tensor = torch::Tensor;

public:
  PtsSampler(const YAML::Node & config);
  SampleResultFlex GetSamples(const Tensor & rays_o, const Tensor & rays_d, RunningMode mode);
  const YAML::Node config_;

  float sample_l_;
};
