//
// Created by ppwang on 2022/9/16.
//

#pragma once
#include "../Utils/Pipe.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class Shader : public Pipe
{
  using Tensor = torch::Tensor;

public:
  virtual Tensor Query(const Tensor & feats, const Tensor & dirs)
  {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  const YAML::Node config;

  int d_in_, d_out_;
};
