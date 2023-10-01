//
// Created by ppwang on 2022/5/8.
//

#ifndef F2_NERF__COMMON_HPP_
#define F2_NERF__COMMON_HPP_

#include <torch/torch.h>

#define None torch::indexing::None
#define Slc torch::indexing::Slice

#define CUDAFloat torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)

#define DivUp(x, y)  (((x) + (y) - 1) / (y))
#define THREAD_CAP 512u
#define LIN_BLOCK_DIM(x) { THREAD_CAP, 1, 1 }
#define LIN_GRID_DIM(x) { unsigned(DivUp((x), THREAD_CAP)), 1, 1 }
#define LINEAR_IDX() (blockIdx.x * blockDim.x + threadIdx.x)
#define RE_INTER(x, y) reinterpret_cast<x>(y)

#endif  // F2_NERF__COMMON_HPP_
