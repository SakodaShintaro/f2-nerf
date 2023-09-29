//
// Created by ppwang on 2022/9/18.
//
#pragma once
#include <vector>
#include <torch/torch.h>

class Pipe : public torch::nn::Module {
  using Tensor = torch::Tensor;

public:
  virtual std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr);
};
