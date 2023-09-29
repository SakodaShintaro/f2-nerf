//
// Created by ppwang on 2022/10/8.
//

#ifndef SANR_SHSHADER_H
#define SANR_SHSHADER_H

#include "../Utils/Pipe.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class SHShader : public Pipe {
  using Tensor = torch::Tensor;
public:
  SHShader(const YAML::Node & config);
  Tensor Query(const Tensor& feats, const Tensor& dirs);
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr) override;

  Tensor SHEncode(const Tensor& dirs);

  torch::nn::Sequential mlp_ = nullptr;

  int degree_;
};


#endif //SANR_SHSHADER_H
