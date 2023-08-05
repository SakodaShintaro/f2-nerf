#include "GridEncoder.hpp"

#include "../Common.h"
#include "../Utils/StopWatch.h"
#include "gridencoder.h"

#include <torch/torch.h>

using Tensor = torch::Tensor;

constexpr bool kAlignCorners = false;
constexpr int64_t kInputDim = 3;

GridEncoder::GridEncoder(const YAML::Node & root_config)
{
  ScopeWatch dataset_watch("GridEncoder::GridEncoder");
  const YAML::Node & config = root_config["field"];

  int64_t num_levels = 16;
  int64_t level_dim = 2;
  per_level_scale = 2;
  base_resolution = 16;
  int64_t log2_hashmap_size = 19;
  double init_std = 1e-4;
  this->interp_id = 0;

  output_dim = num_levels * level_dim;

  std::vector<int64_t> resolutions;
  std::vector<int64_t> offsets;
  int64_t offset = 0;
  max_params = std::pow(2, log2_hashmap_size);
  for (int64_t i = 0; i < num_levels; i++) {
    int64_t resolution =
      static_cast<int64_t>(std::ceil(base_resolution * std::pow(per_level_scale, i)));
    resolution = (kAlignCorners ? resolution : resolution + 1);
    int64_t params_in_level = std::min(
      max_params, static_cast<int64_t>(std::pow(resolution, kInputDim)));        // limit max number
    params_in_level = static_cast<int64_t>(std::ceil(params_in_level / 8) * 8);  // make divisible
    resolutions.push_back(resolution);
    offsets.push_back(offset);
    offset += params_in_level;
  }
  offsets.push_back(offset);
  offsets_ = torch::tensor(offsets).to(torch::kCUDA).to(torch::kInt);
  torch::Tensor idx = torch::empty({offset}, torch::kLong);
  for (int64_t i = 0; i < num_levels; i++) {
    idx.slice(0, offsets[i], offsets[i + 1]) = i;
  }

  n_params = offsets[offsets.size() - 1] * level_dim;

  // Resize the embeddings tensor with the new size
  embeddings_ = torch::empty({offset, level_dim});
  embeddings_.data().uniform_(-init_std, init_std);
  embeddings_ = embeddings_.to(torch::kCUDA);

  // MLP
  int mlp_hidden_dim_, mlp_out_dim_, n_hidden_layers_;
  mlp_hidden_dim_ = config["mlp_hidden_dim"].as<int>();
  mlp_out_dim_ = config["mlp_out_dim"].as<int>();
  n_hidden_layers_ = config["n_hidden_layers"].as<int>();
  mlp_ = std::make_unique<TCNNWP>(
    config, num_levels * level_dim, mlp_out_dim_, mlp_hidden_dim_, n_hidden_layers_);
}

Tensor GridEncoder::Query(torch::Tensor inputs, double bound)
{
#ifdef PROFILE
  ScopeWatch watch(__func__);
#endif

  inputs = (inputs + bound) / (2 * bound);  // map to [0, 1]

  // Compute size before the last dimension
  auto prefix_shape = inputs.sizes().vec();
  prefix_shape.pop_back();
  inputs = inputs.view({-1, kInputDim});

  inputs.requires_grad_(true);

  // "grid_encode" function call would go here... assuming it's some kind of external function
  constexpr int32_t grid_type_id = 0;
  torch::Tensor outputs = torch::autograd::GridEncoderFunction::apply(
    inputs, embeddings_, offsets_, per_level_scale, base_resolution, inputs.requires_grad(),
    grid_type_id, interp_id)[0];
  prefix_shape.push_back(output_dim);
  outputs = outputs.view(prefix_shape);

  outputs = mlp_->Query(outputs);
  return outputs;
}

int GridEncoder::LoadStates(const std::vector<Tensor> & states, int idx)
{
  embeddings_ = states[idx++].clone().to(torch::kCUDA).contiguous();
  mlp_->params_.data().copy_(states[idx++]);
  return idx;
}

std::vector<Tensor> GridEncoder::States()
{
  std::vector<Tensor> ret;
  ret.push_back(embeddings_.data());
  ret.push_back(mlp_->params_.data());
  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> GridEncoder::OptimParamGroups(float lr)
{
  std::vector<torch::optim::OptimizerParamGroup> ret;

  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;

    std::vector<Tensor> params = {embeddings_};
    ret.emplace_back(std::move(params), std::move(opt));
  }

  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;
    opt->weight_decay() = 1e-6;

    std::vector<Tensor> params;
    params.push_back(mlp_->params_);
    ret.emplace_back(std::move(params), std::move(opt));
  }

  return ret;
}

namespace torch::autograd
{

variable_list GridEncoderFunction::forward(
  AutogradContext * ctx, torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
  float per_level_scale, float base_resolution, bool calc_grad_inputs, int32_t gridtype,
  int32_t interpolation)
{
  inputs = inputs.contiguous();

  int64_t B = inputs.size(0);            // batch size
  int64_t D = inputs.size(1);            // coord dim
  int64_t L = offsets.size(0) - 1;       // level
  int64_t C = embeddings.size(1);        // embedding dim for each level
  float S = std::log2(per_level_scale);  // resolution multiplier at each level
  float H = base_resolution;

  torch::Tensor outputs = torch::empty({L, B, C}).to(inputs.device()).to(embeddings.dtype());

  torch::Tensor dy_dx;
  if (calc_grad_inputs) {
    dy_dx = torch::empty({B, L * D * C}).to(inputs.device()).to(embeddings.dtype());
  }

  grid_encode_forward(
    inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, kAlignCorners,
    interpolation);

  outputs = outputs.permute({1, 0, 2}).reshape({B, L * C});

  ctx->save_for_backward({inputs, embeddings, offsets, dy_dx});
  std::vector<int64_t> dims;
  dims.push_back(B);
  dims.push_back(D);
  dims.push_back(C);
  dims.push_back(L);
  dims.push_back(S);
  dims.push_back(H);
  dims.push_back(gridtype);
  dims.push_back(interpolation);
  ctx->saved_data["dims"] = torch::IValue(dims);

  return {outputs};
}

variable_list GridEncoderFunction::backward(AutogradContext * ctx, variable_list grad_output)
{
  variable_list list = ctx->get_saved_variables();
  torch::Tensor inputs = list[0];
  torch::Tensor embeddings = list[1];
  torch::Tensor offsets = list[2];
  torch::Tensor dy_dx = list[3];

  auto dim_list = ctx->saved_data["dims"].toIntList();
  const int64_t B = dim_list[0];
  const int64_t D = dim_list[1];
  const int64_t C = dim_list[2];
  const int64_t L = dim_list[3];
  const int64_t S = dim_list[4];
  const int64_t H = dim_list[5];
  const int64_t gridtype = dim_list[6];
  const int64_t interpolation = dim_list[7];

  // grad: [B, L * C] --> [L, B, C]
  torch::Tensor grad = grad_output[0];
  grad = grad.view({B, L, C}).permute({1, 0, 2}).contiguous();

  torch::Tensor grad_embeddings = torch::zeros_like(embeddings);

  torch::Tensor grad_inputs;
  if (dy_dx.defined()) {
    grad_inputs = torch::zeros_like(inputs).to(embeddings.dtype());
  }

  grid_encode_backward(
    grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs,
    gridtype, kAlignCorners, interpolation);

  if (dy_dx.defined()) {
    grad_inputs = grad_inputs.to(inputs.dtype());
  }

  return {grad_inputs, grad_embeddings, Tensor(), Tensor(), Tensor(),
          Tensor(),    Tensor(),        Tensor(), Tensor()};
}

}  // namespace torch::autograd
