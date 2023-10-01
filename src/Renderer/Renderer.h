//
// Created by ppwang on 2022/5/7.
//

#ifndef SANR_RENDERER_H
#define SANR_RENDERER_H

#include "../Shader/Shader.h"
#include "../field/hash_3d_anchored.hpp"
#include "../points_sampler/points_sampler.hpp"

#include <memory>
#include <vector>

struct RenderResult {
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
  RenderResult Render(const Tensor& rays_o, const Tensor& rays_d, const Tensor& emb_idx, RunningMode mode);

  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr);

  std::shared_ptr<PtsSampler> pts_sampler_;
  std::shared_ptr<Hash3DAnchored> scene_field_;
  std::shared_ptr<Shader> shader_;

  const bool use_app_emb_;
  Tensor app_emb_;
};

#endif //SANR_RENDERER_H
