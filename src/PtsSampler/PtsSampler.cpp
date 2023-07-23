//
// Created by ppwang on 2022/9/26.
//

#include "PtsSampler.h"

#include "../Dataset/Dataset.h"
#include "../Utils/StopWatch.h"
#include "../Utils/Utils.h"

#include <algorithm>
#include <random>

using Tensor = torch::Tensor;

PtsSampler::PtsSampler(const YAML::Node & root_config)
{
  ScopeWatch watch("PtsSampler::PtsSampler");
  const YAML::Node & config = root_config["pts_sampler"];
  sample_l_ = config["sample_l"].as<float>();
}

SampleResultFlex PtsSampler::GetSamples(
  const Tensor & rays_o_raw, const Tensor & rays_d_raw, const Tensor & bounds_raw, RunningMode mode)
{
  Tensor rays_o = rays_o_raw.contiguous();
  Tensor rays_d = (rays_d_raw / torch::linalg_norm(rays_d_raw, 2, -1, true)).contiguous();

  int n_rays = rays_o.sizes()[0];

  // do ray marching
  constexpr int MAX_SAMPLE_PER_RAY = 1024;
  const int n_all_pts = n_rays * MAX_SAMPLE_PER_RAY;

  Tensor rays_noise;
  if (mode == RunningMode::VALIDATE) {
    rays_noise = torch::ones({n_all_pts}, CUDAFloat);
  } else {
    rays_noise = ((torch::rand({n_all_pts}, CUDAFloat) - .5f) + 1.f).contiguous();
  }
  rays_noise.mul_(ray_march_fineness_);
  rays_noise = rays_noise.view({n_rays, MAX_SAMPLE_PER_RAY}).contiguous();
  Tensor cum_noise = torch::cumsum(rays_noise, 1) * sample_l_;
  Tensor sampled_t = cum_noise.reshape({n_all_pts}).contiguous();

  rays_o = rays_o.view({n_rays, 1, 3}).contiguous();
  rays_d = rays_d.view({n_rays, 1, 3}).contiguous();
  cum_noise = cum_noise.unsqueeze(-1).contiguous();
  Tensor sampled_pts = rays_o + rays_d * cum_noise;

  Tensor sampled_dists = torch::diff(sampled_pts, 1, 1).norm(2, -1).contiguous();
  sampled_dists = torch::cat({torch::zeros({n_rays, 1}, CUDAFloat), sampled_dists}, 1).contiguous();
  sampled_pts = sampled_pts.view({n_all_pts, 3});
  sampled_dists = sampled_dists.view({n_all_pts}).contiguous();

  Tensor pts_idx_start_end = torch::ones({n_rays, 2}, CUDAInt) * MAX_SAMPLE_PER_RAY;
  Tensor pts_num = pts_idx_start_end.index({Slc(), 0});
  Tensor cum_num = torch::cumsum(pts_num, 0);
  pts_idx_start_end.index_put_({Slc(), 0}, cum_num - pts_num);
  pts_idx_start_end.index_put_({Slc(), 1}, cum_num);

  Tensor sampled_dirs =
    rays_d.expand({-1, MAX_SAMPLE_PER_RAY, -1}).reshape({n_all_pts, 3}).contiguous();
  Tensor sampled_anchors = torch::zeros({n_all_pts, 3}, CUDAInt);
  Tensor first_oct_dis = torch::full({n_rays, 1}, 1e9f, CUDAFloat).contiguous();

  return {
    sampled_pts,     sampled_dirs,      sampled_dists, sampled_t,
    sampled_anchors, pts_idx_start_end, first_oct_dis,
  };
}
