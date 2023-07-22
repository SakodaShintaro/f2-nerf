//
// Created by ppwang on 2022/9/16.
//

#pragma once
#include "Field.h"

#include <memory>

std::unique_ptr<Field> ConstructField(const YAML::Node & config);
