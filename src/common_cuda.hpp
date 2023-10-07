//
// Created by ppwang on 2022/5/8.
//

#ifndef F2_NERF__COMMON_CUDA_HPP_
#define F2_NERF__COMMON_CUDA_HPP_

#include <cuda_runtime.h>

inline int DivUp(const int x, const int y)
{
  return (x + y - 1) / y;
}

constexpr int THREAD_CAP = 512;
constexpr dim3 LIN_BLOCK_DIM = {unsigned(THREAD_CAP), 1, 1};

inline dim3 LIN_GRID_DIM(const int x)
{
  return dim3{unsigned(DivUp(x, THREAD_CAP)), 1, 1};
}

#endif  // F2_NERF__COMMON_CUDA_HPP_
