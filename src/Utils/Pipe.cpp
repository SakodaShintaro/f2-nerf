//
// Created by ppwang on 2022/9/18.
//

#include "Pipe.h"

using Tensor = torch::Tensor;

std::vector<torch::optim::OptimizerParamGroup> Pipe::OptimParamGroups(float lr)
{
  std::vector<torch::optim::OptimizerParamGroup> ret;
  for (auto sub_module : modules(false)) {
    auto pipe = dynamic_cast<Pipe *>(sub_module.get());
    auto cur_params = pipe->OptimParamGroups(lr);
    for (const auto& para_group : cur_params) {
      ret.emplace_back(para_group);
    }
  }
  return ret;
}
