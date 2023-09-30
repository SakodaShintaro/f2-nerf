//
// Created by ppwang on 2022/5/7.
//

#ifndef SANR_RENDERER_H
#define SANR_RENDERER_H

#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "../Field/Field.h"
#include "../Shader/Shader.h"
#include "../PtsSampler/PtsSampler.h"

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
  std::shared_ptr<Field> scene_field_;
  std::shared_ptr<Shader> shader_;

  const bool use_app_emb_;
  Tensor app_emb_;
};

#endif //SANR_RENDERER_H
