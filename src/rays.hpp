#ifndef F2_NERF__RAY_HPP_
#define F2_NERF__RAY_HPP_

#include <torch/torch.h>

struct alignas(32) Rays
{
  torch::Tensor origins;
  torch::Tensor dirs;
};

Rays get_rays_from_pose(
  const torch::Tensor & pose, const torch::Tensor & intrinsic, const torch::Tensor & ij);

#endif  // F2_NERF__RAY_HPP_
