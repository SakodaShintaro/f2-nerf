//
// Created by ppwang on 2022/10/8.
//

#ifndef SANR_SHSHADER_H
#define SANR_SHSHADER_H

#include <torch/torch.h>

class SHShader : public torch::nn::Module
{
  using Tensor = torch::Tensor;
public:
  SHShader();
  Tensor Query(const Tensor& feats, const Tensor& dirs);
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr);

  Tensor SHEncode(const Tensor& dirs);

  torch::nn::Sequential mlp_ = nullptr;

  const int degree_;
};

#endif //SANR_SHSHADER_H
