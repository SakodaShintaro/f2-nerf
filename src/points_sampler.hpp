//
// Created by ppwang on 2022/6/20.
//

#ifndef F2_NERF__POINTS_SAMPLER_HPP_
#define F2_NERF__POINTS_SAMPLER_HPP_

#include "Eigen/Eigen"
#include "common.hpp"

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

  SampleResultFlex get_samples(const Tensor & rays_o, const Tensor & rays_d, RunningMode mode);

private:
  static constexpr float SAMPLE_L = 1.0 / 256;
};

#endif  // F2_NERF__POINTS_SAMPLER_HPP_
