//
// Created by ppwang on 2022/5/8.
//
#pragma once

#include <torch/torch.h>

#define None torch::indexing::None
#define Slc torch::indexing::Slice

#define CUDAHalf torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
#define CUDAFloat torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
#define CUDALong torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)
#define CUDAInt torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)
#define CUDAUInt8 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)
#define CPUHalf torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU)
#define CPUFloat torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
#define CPULong torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU)
#define CPUInt torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU)

#ifdef HALF_PRECISION
#define CUDAFlex CUDAHalf
#define FlexType __half
#else
#define CUDAFlex CUDAFloat
#define FlexType float
#endif

#define DivUp(x, y)  (((x) + (y) - 1) / (y))
#define THREAD_CAP 512u
#define LIN_BLOCK_DIM(x) { THREAD_CAP, 1, 1 }
#define LIN_GRID_DIM(x) { unsigned(DivUp((x), THREAD_CAP)), 1, 1 }
#define LINEAR_IDX() (blockIdx.x * blockDim.x + threadIdx.x)
#define RE_INTER(x, y) reinterpret_cast<x>(y)
