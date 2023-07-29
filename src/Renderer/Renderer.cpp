//
// Created by ppwang on 2022/5/7.
//

#include "Renderer.h"
#include "../Common.h"
#include "../Utils/Utils.h"
#include "../Utils/StopWatch.h"
#include "../Utils/CustomOps/CustomOps.h"
#include "../Utils/CustomOps/FlexOps.h"
#include "../Utils/CustomOps/Scatter.h"

using Tensor = torch::Tensor;
namespace F = torch::nn::functional;

TORCH_LIBRARY(volume_render, m)
{
  std::cout << "register volume render info" << std::endl;
  m.class_<VolumeRenderInfo>("VolumeRenderInfo").def(torch::init());
}

Renderer::Renderer(const YAML::Node & root_config) : config_(root_config) {
  const YAML::Node conf = root_config["renderer"];

  pts_sampler_ = std::make_unique<PtsSampler>(root_config);
  RegisterSubPipe(pts_sampler_.get());

  scene_field_ = ConstructField(root_config);
  RegisterSubPipe(scene_field_.get());

  shader_ = ConstructShader(root_config);
  RegisterSubPipe(shader_.get());
}

RenderResult Renderer::Render(const Tensor& rays_o, const Tensor& rays_d, RunningMode mode) {
#ifdef PROFILE
  ScopeWatch watch(__func__);
#endif
  int n_rays = rays_o.sizes()[0];
  SampleResultFlex sample_result = pts_sampler_->GetSamples(rays_o, rays_d, mode);
  int n_all_pts = sample_result.pts.sizes()[0];
  float sampled_pts_per_ray = float(n_all_pts) / float(n_rays);
  CHECK(sample_result.pts_idx_bounds.max().item<int>() <= n_all_pts);
  CHECK(sample_result.pts_idx_bounds.min().item<int>() >= 0);

  Tensor bg_color =
    ((mode == RunningMode::TRAIN) ? torch::rand({n_rays, 3}, CUDAFloat)
                                  : torch::ones({n_rays, 3}, CUDAFloat) * .5f);

  if (n_all_pts <= 0) {
    Tensor colors = bg_color;
    return {
      colors,
      torch::zeros({ n_rays, 1 }, CUDAFloat),
      torch::zeros({ n_rays }, CUDAFloat),
      torch::full({ n_rays }, 512.f, CUDAFloat),
      Tensor()
    };
  }
  CHECK_EQ(rays_o.sizes()[0], sample_result.pts_idx_bounds.sizes()[0]);

  auto DensityAct = [](Tensor x) -> Tensor {
    const float shift = 3.f;
    return torch::autograd::TruncExp::apply(x - shift)[0];
  };

  // First, inference without gradients - early stop
  SampleResultFlex sample_result_early_stop;
  {
    Tensor pts  = sample_result.pts;
    Tensor dirs = sample_result.dirs;

    Tensor scene_feat = scene_field_->Query(pts);
    Tensor sampled_density = DensityAct(scene_feat.index({ Slc(), Slc(0, 1) }));

    Tensor sampled_dt = sample_result.dt;
    Tensor sampled_t = (sample_result.t + 1e-2f).contiguous();
    Tensor sec_density = sampled_density.index({Slc(), 0}) * sampled_dt;
    Tensor alphas = 1.f - torch::exp(-sec_density);
    Tensor idx_start_end = sample_result.pts_idx_bounds;
    Tensor acc_density = FlexOps::AccumulateSum(sec_density, idx_start_end, false);
    Tensor trans = torch::exp(-acc_density);
    Tensor weights = trans * alphas;
    Tensor mask = trans > 1e-4f;
    Tensor mask_idx = torch::where(mask)[0];

    sample_result_early_stop.pts = sample_result.pts.index({mask_idx}).contiguous();
    sample_result_early_stop.dirs = sample_result.dirs.index({mask_idx}).contiguous();
    sample_result_early_stop.dt = sample_result.dt.index({mask_idx}).contiguous();
    sample_result_early_stop.t = sample_result.t.index({mask_idx}).contiguous();

    sample_result_early_stop.pts_idx_bounds = FilterIdxBounds(sample_result.pts_idx_bounds, mask);

    CHECK_EQ(sample_result_early_stop.pts_idx_bounds.max().item<int>(), sample_result_early_stop.pts.size(0));
  }

  Tensor pts  = sample_result_early_stop.pts;
  Tensor dirs = sample_result_early_stop.dirs;
  n_all_pts = pts.size(0);

  Tensor scene_feat = scene_field_->Query(pts);

  Tensor sampled_density = DensityAct(scene_feat.index({ Slc(), Slc(0, 1) }));

  Tensor shading_feat = torch::cat({
    torch::ones_like(scene_feat.index({Slc(), Slc(0, 1)}), CUDAFloat),
    scene_feat.index({Slc(), Slc(1, None)})}, 1);

  Tensor sampled_colors = shader_->Query(shading_feat, dirs);

  Tensor sampled_dt = sample_result_early_stop.dt;
  Tensor sampled_t = (sample_result_early_stop.t + 1e-2f).contiguous();
  Tensor sec_density = sampled_density.index({Slc(), 0}) * sampled_dt;
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

  CHECK_NOT_NAN(colors);

  return { colors, disparity, depth, weights, idx_start_end };
}


int Renderer::LoadStates(const std::vector<Tensor>& states, int idx) {
  for (auto pipe : sub_pipes_) {
    idx = pipe->LoadStates(states, idx);
  }

  return idx;
}

std::vector<Tensor> Renderer::States() {
  std::vector<Tensor> ret;
  for (auto pipe : sub_pipes_) {
    auto cur_states = pipe->States();
    ret.insert(ret.end(), cur_states.begin(), cur_states.end());
  }

  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> Renderer::OptimParamGroups(float lr) {
  std::vector<torch::optim::OptimizerParamGroup> ret;
  for (auto pipe : sub_pipes_) {
    auto cur_params = pipe->OptimParamGroups(lr);
    for (const auto& para_group : cur_params) {
      ret.emplace_back(para_group);
    }
  }
  return ret;
}
