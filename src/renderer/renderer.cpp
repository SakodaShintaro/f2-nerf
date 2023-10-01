//
// Created by ppwang on 2022/5/7.
//

#include "renderer.hpp"

#include "../Common.h"
#include "../utils/CustomOps/CustomOps.hpp"
#include "../utils/CustomOps/FlexOps.hpp"
#include "../utils/CustomOps/Scatter.hpp"
#include "../utils/stop_watch.hpp"
#include "../utils/utils.hpp"

using Tensor = torch::Tensor;
namespace F = torch::nn::functional;

Renderer::Renderer(bool use_app_emb, int n_images) : use_app_emb_(use_app_emb)
{
  pts_sampler_ = std::make_shared<PtsSampler>();

  scene_field_ = std::make_shared<Hash3DAnchored>();
  register_module("scene_field", scene_field_);

  shader_ = std::make_shared<SHShader>();
  register_module("shader", shader_);

  // WARNING: Hard code here.
  app_emb_ = torch::randn({n_images, 16}, CUDAFloat) * .1f;
  app_emb_.requires_grad_(true);
  register_parameter("app_emb", app_emb_);
}

RenderResult Renderer::Render(
  const Tensor & rays_o, const Tensor & rays_d, const Tensor & emb_idx, RunningMode mode)
{
  int n_rays = rays_o.sizes()[0];
  SampleResultFlex sample_result = pts_sampler_->GetSamples(rays_o, rays_d, mode);
  int n_all_pts = sample_result.pts.sizes()[0];
  CHECK(sample_result.pts_idx_bounds.max().item<int>() <= n_all_pts);
  CHECK(sample_result.pts_idx_bounds.min().item<int>() >= 0);

  Tensor bg_color =
    ((mode == RunningMode::TRAIN) ? torch::rand({n_rays, 3}, CUDAFloat)
                                  : torch::ones({n_rays, 3}, CUDAFloat) * .5f);

  if (n_all_pts <= 0) {
    return {
      bg_color, torch::zeros({n_rays, 1}, CUDAFloat), torch::zeros({n_rays}, CUDAFloat),
      torch::full({n_rays}, 512.f, CUDAFloat), Tensor()};
  }
  CHECK(rays_o.sizes()[0] == sample_result.pts_idx_bounds.sizes()[0]);

  auto DensityAct = [](Tensor x) -> Tensor {
    const float shift = 3.f;
    return torch::autograd::TruncExp::apply(x - shift)[0];
  };

  // First, inference - early stop
  SampleResultFlex sample_result_early_stop;
  {
    Tensor scene_feat = scene_field_->Query(sample_result.pts);
    Tensor sampled_density = DensityAct(scene_feat.index({Slc(), Slc(0, 1)}));
    Tensor sec_density = sampled_density.index({Slc(), 0}) * sample_result.dt;
    Tensor alphas = 1.f - torch::exp(-sec_density);
    Tensor acc_density = FlexOps::AccumulateSum(sec_density, sample_result.pts_idx_bounds, false);
    Tensor trans = torch::exp(-acc_density);
    Tensor weights = trans * alphas;
    Tensor mask = trans > 1e-4f;
    Tensor mask_idx = torch::where(mask)[0];

    sample_result_early_stop.pts = sample_result.pts.index({mask_idx}).contiguous();
    sample_result_early_stop.dirs = sample_result.dirs.index({mask_idx}).contiguous();
    sample_result_early_stop.dt = sample_result.dt.index({mask_idx}).contiguous();
    sample_result_early_stop.t = sample_result.t.index({mask_idx}).contiguous();

    Tensor mask_2d = mask.reshape({n_rays, MAX_SAMPLE_PER_RAY});
    Tensor num = mask_2d.sum(1);
    Tensor cum_num = torch::cumsum(num, 0);
    Tensor idx_bounds =
      torch::zeros({n_rays, 2}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    idx_bounds.index_put_({Slc(), 0}, cum_num - num);
    idx_bounds.index_put_({Slc(), 1}, cum_num);
    sample_result_early_stop.pts_idx_bounds = idx_bounds;

    CHECK(
      sample_result_early_stop.pts_idx_bounds.max().item<int>() ==
      sample_result_early_stop.pts.size(0));
  }

  n_all_pts = sample_result_early_stop.pts.size(0);

  Tensor scene_feat = scene_field_->Query(sample_result_early_stop.pts);
  Tensor sampled_density = DensityAct(scene_feat.index({Slc(), Slc(0, 1)}));

  Tensor shading_feat = torch::cat(
    {torch::ones_like(scene_feat.index({Slc(), Slc(0, 1)}), CUDAFloat),
     scene_feat.index({Slc(), Slc(1, torch::indexing::None)})},
    1);

  if (mode == RunningMode::TRAIN && use_app_emb_) {
    Tensor all_emb_idx =
      CustomOps::ScatterIdx(n_all_pts, sample_result_early_stop.pts_idx_bounds, emb_idx);
    shading_feat = CustomOps::ScatterAdd(app_emb_, all_emb_idx, shading_feat);
  }

  Tensor sampled_colors = shader_->Query(shading_feat, sample_result_early_stop.dirs);
  Tensor sampled_t = (sample_result_early_stop.t + 1e-2f).contiguous();
  Tensor sec_density = sampled_density.index({Slc(), 0}) * sample_result_early_stop.dt;
  Tensor alphas = 1.f - torch::exp(-sec_density);
  Tensor idx_start_end = sample_result_early_stop.pts_idx_bounds;
  Tensor acc_density = FlexOps::AccumulateSum(sec_density, idx_start_end, false);
  Tensor trans = torch::exp(-acc_density);
  Tensor weights = trans * alphas;

  Tensor last_trans = torch::exp(-FlexOps::Sum(sec_density, idx_start_end));
  Tensor colors = FlexOps::Sum(weights.unsqueeze(-1) * sampled_colors, idx_start_end);
  colors = colors + last_trans.unsqueeze(-1) * bg_color;
  Tensor disparity = FlexOps::Sum(weights / sampled_t, idx_start_end);
  Tensor depth = FlexOps::Sum(weights * sampled_t, idx_start_end) / (1.f - last_trans + 1e-4f);

  CHECK(std::isfinite((colors).mean().item<float>()));

  return {colors, disparity, depth, weights, idx_start_end};
}

std::vector<torch::optim::OptimizerParamGroup> Renderer::OptimParamGroups(float lr)
{
  std::vector<torch::optim::OptimizerParamGroup> ret;

  // scene_field_
  for (const auto & para_group : scene_field_->OptimParamGroups(lr)) {
    ret.emplace_back(para_group);
  }

  // shader_
  for (const auto & para_group : shader_->OptimParamGroups(lr)) {
    ret.emplace_back(para_group);
  }

  // app_emb_
  auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
  opt->betas() = {0.9, 0.99};
  opt->eps() = 1e-15;
  opt->weight_decay() = 1e-6;
  std::vector<Tensor> params{app_emb_};
  ret.emplace_back(std::move(params), std::move(opt));

  return ret;
}
