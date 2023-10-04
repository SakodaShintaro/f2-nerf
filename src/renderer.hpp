//
// Created by ppwang on 2022/5/7.
//

#ifndef F2_NERF__RENDERER_HPP_
#define F2_NERF__RENDERER_HPP_

#include "hash_3d_anchored.hpp"
#include "points_sampler.hpp"
#include "sh_shader.hpp"

#include <memory>
#include <vector>

struct RenderResult
{
  using Tensor = torch::Tensor;
  Tensor colors;
  Tensor disparity;
  Tensor depth;
  Tensor weights;
  Tensor idx_start_end;
};

class Renderer : public torch::nn::Module
{
  using Tensor = torch::Tensor;

public:
  Renderer(bool use_app_emb, int n_images);

  RenderResult render(
    const Tensor & rays_o, const Tensor & rays_d, const Tensor & emb_idx, RunningMode mode);

  std::tuple<Tensor, Tensor> render_all_rays(const Tensor & rays_o, const Tensor & rays_d);

  std::vector<torch::optim::OptimizerParamGroup> optim_param_groups(float lr);

private:
  std::shared_ptr<PtsSampler> pts_sampler_;
  std::shared_ptr<Hash3DAnchored> scene_field_;
  std::shared_ptr<SHShader> shader_;

  const bool use_app_emb_;
  Tensor app_emb_;
};

#endif  // F2_NERF__RENDERER_HPP_
