//
// Created by ppwang on 2022/9/16.
//

#include "ShaderFactory.h"

#include "SHShader.h"

std::unique_ptr<Shader> ConstructShader(const YAML::Node & config)
{
  auto type = config["shader"]["type"].as<std::string>();
  if (type == "SHShader") {
    return std::make_unique<SHShader>(config);
  } else {
    CHECK(false) << "There is no such shader type.";
    return {};
  }
}
