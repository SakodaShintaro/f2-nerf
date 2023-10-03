//
// Created by ppwang on 2022/5/11.
//

#ifndef F2_NERF__UTILS_HPP_
#define F2_NERF__UTILS_HPP_

#include <torch/torch.h>

#include <string>

namespace utils
{
using Tensor = torch::Tensor;

Tensor read_image_tensor(const std::string & path);
bool write_image_tensor(const std::string & path, Tensor img);

}  // namespace utils

#endif  // F2_NERF__UTILS_HPP_
