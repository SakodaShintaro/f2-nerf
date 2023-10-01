//
// Created by ppwang on 2022/5/11.
//

#ifndef F2_NERF__UTILS_HPP_
#define F2_NERF__UTILS_HPP_

#include <string>
#include <torch/torch.h>

namespace Utils {
using Tensor = torch::Tensor;

Tensor ReadImageTensor(const std::string& path);
bool WriteImageTensor(const std::string& path, Tensor img);

}

#endif  // F2_NERF__UTILS_HPP_
