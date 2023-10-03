//
// Created by ppwang on 2022/7/17.
//
#include "common.hpp"
#include "hash_3d_anchored.hpp"

#include <Eigen/Eigen>

#include <torch/torch.h>

using Tensor = torch::Tensor;

constexpr float RES_FINE_POW_2 = 10.f;
constexpr float RES_BASE_POW_2 = 3.f;

#ifdef HALF_PRECISION
#define CUDAFlex torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
#define FlexType __half
#else
#define CUDAFlex torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
#define FlexType float
#endif

template <typename T>
__global__ void Hash3DAnchoredForwardKernel(
  int n_points, int local_size, T * feat_pool, int * prim_pool, Eigen::Vector3f * bias_pool,
  Eigen::Vector3f * points_ptr, T * out_feat)
{
  int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int level_idx = blockIdx.y;
  if (pts_idx >= n_points) {
    return;
  }

  points_ptr = points_ptr + pts_idx;
  out_feat = out_feat + pts_idx * (N_LEVELS * N_CHANNELS);

  Eigen::Vector3f pt = points_ptr[0];
  float mul = exp2f(
    (RES_FINE_POW_2 - RES_BASE_POW_2) * float(level_idx) / float(N_LEVELS - 1) + RES_BASE_POW_2);
  pt *= mul;
  unsigned prim_a, prim_b, prim_c;
  {
    const int offset = level_idx * 3;
    prim_a = prim_pool[offset + 0];
    prim_b = prim_pool[offset + 1];
    prim_c = prim_pool[offset + 2];
  }
  feat_pool = feat_pool + local_size * level_idx;

  pt = pt + bias_pool[level_idx];

  auto pos_x = static_cast<unsigned>(floorf(pt[0]));
  auto pos_y = static_cast<unsigned>(floorf(pt[1]));
  auto pos_z = static_cast<unsigned>(floorf(pt[2]));

  unsigned pos_000 = ((pos_x * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_001 = ((pos_x * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
  unsigned pos_010 = ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_011 =
    ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
  unsigned pos_100 = (((pos_x + 1u) * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_101 =
    (((pos_x + 1u) * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
  unsigned pos_110 =
    (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_111 =
    (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;

  float a = pt[0] - floorf(pt[0]);
  float b = pt[1] - floorf(pt[1]);
  float c = pt[2] - floorf(pt[2]);

  float w000 = (1.f - a) * (1.f - b) * (1.f - c);
  float w001 = (1.f - a) * (1.f - b) * c;
  float w010 = (1.f - a) * b * (1.f - c);
  float w011 = (1.f - a) * b * c;
  float w100 = a * (1.f - b) * (1.f - c);
  float w101 = a * (1.f - b) * c;
  float w110 = a * b * (1.f - c);
  float w111 = a * b * c;

#pragma unroll
  for (int k = 0; k < N_CHANNELS; k++) {
    out_feat[level_idx * N_CHANNELS + k] = (T)(w000 * float(feat_pool[pos_000 * N_CHANNELS + k]) +
                                               w001 * float(feat_pool[pos_001 * N_CHANNELS + k]) +
                                               w010 * float(feat_pool[pos_010 * N_CHANNELS + k]) +
                                               w011 * float(feat_pool[pos_011 * N_CHANNELS + k]) +
                                               w100 * float(feat_pool[pos_100 * N_CHANNELS + k]) +
                                               w101 * float(feat_pool[pos_101 * N_CHANNELS + k]) +
                                               w110 * float(feat_pool[pos_110 * N_CHANNELS + k]) +
                                               w111 * float(feat_pool[pos_111 * N_CHANNELS + k]));
  }
}

template <typename T>
__global__ void Hash3DAnchoredBackwardKernel(
  int n_points, int local_size, T * feat_pool, int * prim_pool, Eigen::Vector3f * bias_pool,
  Eigen::Vector3f * points_ptr,
  T * grad_in,      // [ n_points, n_levels, n_channels ]
  T * grad_points,  // [ n_points, 3 ]
  T * grad_embeds   // [ pool_size, n_channels ]
)
{
  int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int level_idx = blockIdx.y;
  if (pts_idx >= n_points) {
    return;
  }
  feat_pool = feat_pool + local_size * level_idx;
  points_ptr = points_ptr + pts_idx;
  grad_in = grad_in + (N_LEVELS * N_CHANNELS) * pts_idx + level_idx * N_CHANNELS;
  Eigen::Vector3f pt = points_ptr[0];
  float mul = exp2f(
    (RES_FINE_POW_2 - RES_BASE_POW_2) * float(level_idx) / float(N_LEVELS - 1) + RES_BASE_POW_2);
  pt *= mul;
  unsigned prim_a, prim_b, prim_c;
  {
    const int offset = level_idx * 3;
    prim_a = prim_pool[offset + 0];
    prim_b = prim_pool[offset + 1];
    prim_c = prim_pool[offset + 2];
  }

  grad_points = grad_points + pts_idx * 3;
  grad_embeds = grad_embeds + local_size * level_idx;

  pt = pt + bias_pool[level_idx];

  unsigned pos_x = static_cast<unsigned>(floorf(pt[0]));
  unsigned pos_y = static_cast<unsigned>(floorf(pt[1]));
  unsigned pos_z = static_cast<unsigned>(floorf(pt[2]));

  unsigned pos_000 = ((pos_x * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_001 = ((pos_x * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
  unsigned pos_010 = ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_011 =
    ((pos_x * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
  unsigned pos_100 = (((pos_x + 1u) * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_101 =
    (((pos_x + 1u) * prim_a) ^ ((pos_y)*prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;
  unsigned pos_110 =
    (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z)*prim_c)) % local_size;
  unsigned pos_111 =
    (((pos_x + 1u) * prim_a) ^ ((pos_y + 1u) * prim_b) ^ ((pos_z + 1u) * prim_c)) % local_size;

  float a = pt[0] - floorf(pt[0]);
  float b = pt[1] - floorf(pt[1]);
  float c = pt[2] - floorf(pt[2]);

  float w000 = (1.f - a) * (1.f - b) * (1.f - c);
  float w001 = (1.f - a) * (1.f - b) * c;
  float w010 = (1.f - a) * b * (1.f - c);
  float w011 = (1.f - a) * b * c;
  float w100 = a * (1.f - b) * (1.f - c);
  float w101 = a * (1.f - b) * c;
  float w110 = a * b * (1.f - c);
  float w111 = a * b * c;

  float ws[8] = {w000, w001, w010, w011, w100, w101, w110, w111};
  unsigned pos[8] = {pos_000, pos_001, pos_010, pos_011, pos_100, pos_101, pos_110, pos_111};
  float sign_x[8] = {-1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float sign_y[8] = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f};
  float sign_z[8] = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

#pragma unroll
  for (int d = 0; d < 8; d++) {
    for (int k = 0; k < N_CHANNELS; k += 2) {
      float w0 = (float)grad_in[k];
      float w1 = (float)grad_in[k + 1];
      if (w0 != 0.f || w1 != 0.f) {
        __half2 cur_w = {(__half)(float(w0) * ws[d]), (__half)(float(w1) * ws[d])};
        atomicAdd((__half2 *)(grad_embeds + pos[d] * N_CHANNELS + k), cur_w);
      }
    }
    atomicAdd(
      grad_points + 0,
      (T)(sign_x[d] * (float)(feat_pool[pos[d] * N_CHANNELS + 0]) * mul * (float)grad_in[0]));
    atomicAdd(
      grad_points + 1,
      (T)(sign_y[d] * (float)(feat_pool[pos[d] * N_CHANNELS + 0]) * mul * (float)grad_in[0]));
    atomicAdd(
      grad_points + 2,
      (T)(sign_z[d] * (float)(feat_pool[pos[d] * N_CHANNELS + 0]) * mul * (float)grad_in[0]));
    atomicAdd(
      grad_points + 0,
      (T)(sign_x[d] * (float)(feat_pool[pos[d] * N_CHANNELS + 1]) * mul * (float)grad_in[1]));
    atomicAdd(
      grad_points + 1,
      (T)(sign_y[d] * (float)(feat_pool[pos[d] * N_CHANNELS + 1]) * mul * (float)grad_in[1]));
    atomicAdd(
      grad_points + 2,
      (T)(sign_z[d] * (float)(feat_pool[pos[d] * N_CHANNELS + 1]) * mul * (float)grad_in[1]));
  }
}

namespace torch::autograd
{

variable_list Hash3DAnchoredFunction::forward(
  AutogradContext * ctx, Tensor points, Tensor feat_pool, IValue hash3d_info)
{
  auto info_ptr = hash3d_info.toCustomClass<Hash3DAnchoredInfo>();
  ctx->saved_data["hash3d_info"] = hash3d_info;
  ctx->saved_data["points"] = points;
  ctx->saved_data["feat_pool"] = feat_pool;
  Tensor & prim_pool = info_ptr->hash3d_->prim_pool_;
  Tensor & bias_pool = info_ptr->hash3d_->bias_pool_;
  CHECK(points.device().is_cuda());

  int n_points = points.sizes()[0];

  const unsigned thread_cap = 512;
  dim3 block_dim = {unsigned(thread_cap), 1, 1};
  dim3 grid_dim = {DivUp(n_points, thread_cap), unsigned(N_LEVELS), 1};

  Tensor out_feat = torch::zeros({n_points, N_LEVELS * N_CHANNELS}, CUDAFlex);
  CHECK(out_feat.is_contiguous());

  Tensor feat_pool_true = feat_pool.to(torch::kFloat16).contiguous();

  Hash3DAnchoredForwardKernel<FlexType><<<grid_dim, block_dim>>>(
    n_points, info_ptr->hash3d_->local_size_,
    reinterpret_cast<FlexType *>(feat_pool_true.data_ptr()), prim_pool.data_ptr<int>(),
    reinterpret_cast<Eigen::Vector3f *>(bias_pool.data_ptr()),
    reinterpret_cast<Eigen::Vector3f *>(points.data_ptr()),
    reinterpret_cast<FlexType *>(out_feat.data_ptr()));

  return {out_feat.to(torch::kFloat32)};
}

variable_list Hash3DAnchoredFunction::backward(AutogradContext * ctx, variable_list grad_output)
{
  auto info_ptr = ctx->saved_data["hash3d_info"].toCustomClass<Hash3DAnchoredInfo>();
  Tensor & points = ctx->saved_data["points"].toTensor();  // [ n_points, 3 ]
  Tensor & feat_pool = ctx->saved_data["feat_pool"].toTensor();
  Tensor & prim_pool = info_ptr->hash3d_->prim_pool_;
  Tensor & bias_pool = info_ptr->hash3d_->bias_pool_;
  CHECK(points.device().is_cuda());

  const float grad_scale = 128.f;
  int n_points = points.sizes()[0];

  int pool_size = info_ptr->hash3d_->pool_size_;

  const unsigned thread_cap = 512;
  dim3 block_dim = {unsigned(thread_cap), 1, 1};
  dim3 grid_dim = {DivUp(n_points, thread_cap), unsigned(N_LEVELS), 1};

  Tensor feat_pool_true = feat_pool.to(torch::kFloat16).contiguous();

  Tensor grad_in = (grad_output[0] * grad_scale).to(torch::kFloat16).contiguous();

  Tensor points_grad = torch::zeros({n_points, 3}, CUDAFlex);
  Tensor embeds_grad = torch::zeros({pool_size, N_CHANNELS}, CUDAFlex);

  Hash3DAnchoredBackwardKernel<FlexType><<<grid_dim, block_dim>>>(
    n_points, info_ptr->hash3d_->local_size_,
    reinterpret_cast<FlexType *>(feat_pool_true.data_ptr()), prim_pool.data_ptr<int>(),
    reinterpret_cast<Eigen::Vector3f *>(bias_pool.data_ptr()),
    reinterpret_cast<Eigen::Vector3f *>(points.data_ptr()),
    reinterpret_cast<FlexType *>(grad_in.data_ptr()),
    reinterpret_cast<FlexType *>(points_grad.data_ptr()),
    reinterpret_cast<FlexType *>(embeds_grad.data_ptr()));

  points_grad = points_grad.to(torch::kFloat32) / grad_scale;
  embeds_grad = embeds_grad.to(torch::kFloat32) / grad_scale;

  return {points_grad, embeds_grad, Tensor()};
}

}  // namespace torch::autograd
