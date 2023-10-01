//
// Created by ppwang on 2022/7/17.
//

#ifndef F2_NERF__HASH_3D_ANCHORED_HPP_
#define F2_NERF__HASH_3D_ANCHORED_HPP_

#include <torch/torch.h>

#define N_CHANNELS 2
#define N_LEVELS 16
// 1024
#define RES_FINE_POW_2 10.f
// 8
#define RES_BASE_POW_2 3.f

class Hash3DAnchored : public torch::nn::Module
{
  using Tensor = torch::Tensor;

public:
  Hash3DAnchored();

  Tensor Query(const Tensor & points);

  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr);

  int pool_size_;
  int local_size_;

  Tensor feat_pool_;  // [ pool_size_, n_channels_ ];
  Tensor prim_pool_;  // [ n_levels, 3 ];
  Tensor bias_pool_;  // [ n_levels, 3 ];

  torch::nn::Linear mlp_ = nullptr;
};

class Hash3DAnchoredInfo : public torch::CustomClassHolder
{
public:
  Hash3DAnchored * hash3d_ = nullptr;
};

namespace torch::autograd
{

class Hash3DAnchoredFunction : public Function<Hash3DAnchoredFunction>
{
public:
  static variable_list forward(
    AutogradContext * ctx, Tensor points, Tensor feat_pool_, IValue hash3d_info);

  static variable_list backward(AutogradContext * ctx, variable_list grad_output);
};

}  // namespace torch::autograd

#endif  // F2_NERF__HASH_3D_ANCHORED_HPP_
