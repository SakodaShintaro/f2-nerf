//
// Created by ppwang on 2022/9/16.
//
#include "FieldFactory.h"

#include "Hash3DAnchored.h"

std::unique_ptr<Field> ConstructField(const YAML::Node & config)
{
  auto type = config["field"]["type"].as<std::string>();
  if (type == "Hash3DAnchored") {
    return std::make_unique<Hash3DAnchored>(config);
  } else {
    CHECK(false) << "There is no such field type.";
    return {};
  }
}
