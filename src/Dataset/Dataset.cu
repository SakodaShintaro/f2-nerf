//
// Created by ppwang on 2022/9/21.
//

#include "Dataset.h"
#include <torch/torch.h>
#include <Eigen/Eigen>
#include "../Common.h"


using Tensor = torch::Tensor;

__global__ void Img2WorldRayKernel(int n_rays,
                                   Watrix34f* poses,
                                   Watrix33f* intri,
                                   Wec4f* dist_params,
                                   int* cam_indices,
                                   Wec2f* ij,
                                   Wec3f* out_rays_o,
                                   Wec3f* out_rays_d) {
  int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= n_rays) { return; }

  int cam_idx = cam_indices[ray_idx];
  float i = static_cast<float>(ij[ray_idx][0]);
  float j = static_cast<float>(ij[ray_idx][1]);
  float cx = intri[cam_idx](0, 2);
  float cy = intri[cam_idx](1, 2);
  float fx = intri[cam_idx](0, 0);
  float fy = intri[cam_idx](1, 1);

  float u = (j - cx) / fx;
  float v = (i - cy) / fy;  // OpenCV style
  Wec3f dir = {u, -v, -1.f };  // OpenGL style
  out_rays_d[ray_idx] = poses[cam_idx].block<3, 3>(0, 0) * dir;
  out_rays_o[ray_idx] = poses[cam_idx].block<3, 1>(0, 3);
}

Rays Dataset::Img2WorldRayFlex(const Tensor& cam_indices, const Tensor& ij) {
  Tensor ij_shift = (ij + .5f).contiguous();
  CK_CONT(cam_indices);
  CK_CONT(ij_shift);
  CK_CONT(poses_);
  CK_CONT(intri_);
  CK_CONT(dist_params_);
  CHECK_EQ(poses_.sizes()[0], intri_.sizes()[0]);
  CHECK_EQ(cam_indices.sizes()[0], ij.sizes()[0]);

  const int n_rays = cam_indices.sizes()[0];
  dim3 block_dim = LIN_BLOCK_DIM(n_rays);
  dim3 grid_dim  = LIN_GRID_DIM(n_rays);

  Tensor rays_o = torch::zeros({n_rays, 3}, CUDAFloat).contiguous();
  Tensor rays_d = torch::zeros({n_rays, 3}, CUDAFloat).contiguous();

  Img2WorldRayKernel<<<grid_dim, block_dim>>>(n_rays,
                                              RE_INTER(Watrix34f *, poses_.data_ptr()),
                                              RE_INTER(Watrix33f *, intri_.data_ptr()),
                                              RE_INTER(Wec4f*, dist_params_.data_ptr()),
                                              cam_indices.data_ptr<int>(),
                                              RE_INTER(Wec2f*, ij_shift.data_ptr()),
                                              RE_INTER(Wec3f*, rays_o.data_ptr()),
                                              RE_INTER(Wec3f*, rays_d.data_ptr()));

  return { rays_o, rays_d };
}
