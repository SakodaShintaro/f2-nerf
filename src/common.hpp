//
// Created by ppwang on 2022/5/8.
//

#ifndef F2_NERF__COMMON_HPP_
#define F2_NERF__COMMON_HPP_

#include <torch/torch.h>

using Slc = torch::indexing::Slice;
const auto CUDAFloat = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

#endif  // F2_NERF__COMMON_HPP_
