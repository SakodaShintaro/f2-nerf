//
// Created by ppwang on 2022/6/20.
//

#ifndef F2_NERF__POINTS_SAMPLER_HPP_
#define F2_NERF__POINTS_SAMPLER_HPP_

#include "common.hpp"
#include "Eigen/Eigen"

#include <torch/torch.h>

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
  PtsSampler();
  SampleResultFlex GetSamples(const Tensor & rays_o, const Tensor & rays_d, RunningMode mode);

  const float sample_l_;
};

#endif  // F2_NERF__POINTS_SAMPLER_HPP_
