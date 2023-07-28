//
// Created by ppwang on 2022/9/16.
//
#pragma once
#include "../Utils/Pipe.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class Field : public Pipe
{
  using Tensor = torch::Tensor;

public:
  virtual Tensor Query(const Tensor & coords)
  {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  const YAML::Node config_;
};
