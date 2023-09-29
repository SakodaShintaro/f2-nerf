//
// Created by ppwang on 2022/7/17.
//

#include "Hash3DAnchored.h"
#include <torch/torch.h>
#include "../Common.h"
#include "../Utils/StopWatch.h"

using Tensor = torch::Tensor;

TORCH_LIBRARY(dec_hash3d_anchored, m)
{
  std::cout << "register Hash3DAnchoredInfo" << std::endl;
  m.class_<Hash3DAnchoredInfo>("Hash3DAnchoredInfo").def(torch::init());
}

Hash3DAnchored::Hash3DAnchored(const YAML::Node & root_config) : config_(root_config)
{
  ScopeWatch dataset_watch("Hash3DAnchored::Hash3DAnchored");
  const YAML::Node & config = root_config["field"];

  pool_size_ = (1 << config["log2_table_size"].as<int>()) * N_LEVELS;

  // Feat pool
  feat_pool_ = (torch::rand({pool_size_, N_CHANNELS}, CUDAFloat) * .2f - 1.f) * 1e-4f;
  feat_pool_.requires_grad_(true);
  CHECK(feat_pool_.is_contiguous());

  // Get prime numbers
  auto is_prim = [](int x) {
    for (int i = 2; i * i <= x; i++) {
      if (x % i == 0) return false;
    }
    return true;
  };

  std::vector<int> prim_selected;
  int min_local_prim = 1 << 28;
  int max_local_prim = 1 << 30;

  for (int i = 0; i < 3 * N_LEVELS; i++) {
    int val;
    do {
      val = torch::randint(min_local_prim, max_local_prim, {1}, CPUInt).item<int>();
    }
    while (!is_prim(val));
    prim_selected.push_back(val);
  }

  CHECK_EQ(prim_selected.size(), 3 * N_LEVELS);

  prim_pool_ = torch::from_blob(prim_selected.data(), 3 * N_LEVELS, CPUInt).to(torch::kCUDA);
  prim_pool_ = prim_pool_.reshape({N_LEVELS, 3}).contiguous();

  if (config["rand_bias"].as<bool>()) {
    bias_pool_ = (torch::rand({ N_LEVELS, 3 }, CUDAFloat) * 1000.f + 100.f).contiguous();
  }
  else {
    bias_pool_ = torch::zeros({ N_LEVELS, 3 }, CUDAFloat).contiguous();
  }

  local_size_ = pool_size_ / N_LEVELS;
  local_size_ = (local_size_ >> 4) << 4;

  // MLP
  int mlp_out_dim = config["mlp_out_dim"].as<int>();
  mlp_ = torch::nn::Linear(N_LEVELS * N_CHANNELS, mlp_out_dim);

  register_parameter("feat_pool", feat_pool_);
  register_parameter("prim_pool", prim_pool_, false);
  register_parameter("bias_pool", bias_pool_);
  register_module("mlp", mlp_);
}

Tensor Hash3DAnchored::Query(const Tensor& points) {
#ifdef PROFILE
  ScopeWatch watch(__func__);
#endif

  auto info = torch::make_intrusive<Hash3DAnchoredInfo>();
  info->hash3d_ = this;

  const float radius = 1.0f;
  Tensor norm = points.norm(2, {1}, true);
  Tensor mask = (norm <= radius);
  Tensor x = points * mask + ~mask * (1 + radius - radius / norm) * points / norm;

  Tensor feat = torch::autograd::Hash3DAnchoredFunction::apply(x, feat_pool_, torch::IValue(info))[0];
  Tensor output = mlp_->forward(feat);
  return output;
}

std::vector<torch::optim::OptimizerParamGroup> Hash3DAnchored::OptimParamGroups(float lr)
{
  std::vector<torch::optim::OptimizerParamGroup> ret;

  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;

    std::vector<Tensor> params = { feat_pool_ };
    ret.emplace_back(std::move(params), std::move(opt));
  }

  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;
    opt->weight_decay() = 1e-6;

    std::vector<Tensor> params = mlp_->parameters();
    ret.emplace_back(std::move(params), std::move(opt));
  }

  return ret;
}
