//
// Created by ppwang on 2022/10/8.
//

#include "SHShader.h"
#include "../Common.h"

using Tensor = torch::Tensor;

SHShader::SHShader(const YAML::Node & root_config)
{
  const YAML::Node config = root_config["shader"];
  d_in_ = config["d_in"].as<int>();
  d_out_ = config["d_out"].as<int>();
  degree_ = config["degree"].as<int>();
  d_hidden_ = config["d_hidden"].as<int>();
  n_hiddens_ = config["n_hiddens"].as<int>();

  mlp_ = torch::nn::Sequential(
    torch::nn::Linear(d_in_, d_hidden_), torch::nn::ReLU(), torch::nn::Linear(d_hidden_, d_out_));
  register_module("mlp", mlp_);
}

Tensor SHShader::Query(const Tensor &feats, const Tensor &dirs) {
  Tensor enc = SHEncode(dirs);
  Tensor input = torch::cat({ feats, enc }, -1);
  Tensor output = mlp_->forward(input);
  float eps = 1e-3f;
  return (1.f + 2.f * eps) / (1.f + torch::exp(-output)) - eps;
}

std::vector<torch::optim::OptimizerParamGroup> SHShader::OptimParamGroups(float lr) {
  auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
  opt->betas() = { 0.9, 0.99 };
  opt->eps() = 1e-15;
  opt->weight_decay() = 1e-6;

  std::vector<Tensor> params = mlp_->parameters();
  return { torch::optim::OptimizerParamGroup(params, std::move(opt)) };
}
