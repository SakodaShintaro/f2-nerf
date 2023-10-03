//
// Created by ppwang on 2022/10/8.
//

#include "sh_shader.hpp"

#include "common.hpp"

using Tensor = torch::Tensor;

SHShader::SHShader()
{
  const int d_in = 32;
  const int d_hidden = 64;
  const int d_out = 3;

  mlp_ = torch::nn::Sequential(
    torch::nn::Linear(d_in, d_hidden), torch::nn::ReLU(), torch::nn::Linear(d_hidden, d_out));
  register_module("mlp", mlp_);
}

Tensor SHShader::query(const Tensor & feats, const Tensor & dirs)
{
  Tensor enc = encode(dirs);
  Tensor input = torch::cat({feats, enc}, -1);
  Tensor output = mlp_->forward(input);
  float eps = 1e-3f;
  return (1.f + 2.f * eps) / (1.f + torch::exp(-output)) - eps;
}

std::vector<torch::optim::OptimizerParamGroup> SHShader::optim_param_groups(float lr)
{
  auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
  opt->betas() = {0.9, 0.99};
  opt->eps() = 1e-15;
  opt->weight_decay() = 1e-6;

  std::vector<Tensor> params = mlp_->parameters();
  return {torch::optim::OptimizerParamGroup(params, std::move(opt))};
}
