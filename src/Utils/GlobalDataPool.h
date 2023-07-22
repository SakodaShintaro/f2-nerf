//
// Created by ppwang on 2022/9/16.
//
#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

enum RunningMode { TRAIN, VALIDATE };

class GlobalDataPool {
public:
  GlobalDataPool(const std::string& config_path);

  YAML::Node config_;
  RunningMode mode_;

  float learning_rate_ = 1.f;
};

