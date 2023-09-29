//
// Created by ppwang on 2022/5/11.
//

#include <string>
#include <torch/torch.h>

namespace Utils {
using Tensor = torch::Tensor;

Tensor ReadImageTensor(const std::string& path);
bool WriteImageTensor(const std::string& path, Tensor img);

}