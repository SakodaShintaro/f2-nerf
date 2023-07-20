//
// Created by ppwang on 2022/9/26.
//

#include <random>
#include <algorithm>
#include "../Utils/Utils.h"
#include "../Utils/StopWatch.h"
#include "../Dataset/Dataset.h"
#include "PersSampler.h"

using Tensor = torch::Tensor;

namespace {

float DistanceSummary(const Tensor& dis) {
  if (dis.reshape(-1).size(0) <= 0) { return 1e8f; }
  Tensor log_dis = torch::log(dis);
  float thres = torch::quantile(log_dis, 0.25).item<float>();
  Tensor mask = (log_dis < thres).to(torch::kFloat32);
  if (mask.sum().item<float>() < 1e-3f) {
    return std::exp(log_dis.mean().item<float>());
  }
  return std::exp(((log_dis * mask).sum() / mask.sum()).item<float>());
}

std::vector<int> GetVisiCams(float bbox_side_len,
                             const Tensor& center,
                             const Tensor& c2w,
                             const Tensor& intri,
                             const Tensor& bound) {
  float half_w = intri.index({0, 0, 2}).item<float>();
  float half_h = intri.index({0, 1, 2}).item<float>();
  float cx = intri.index({ 0, 0, 2 }).item<float>();
  float cy = intri.index({ 0, 1, 2 }).item<float>();
  float fx = intri.index({ 0, 0, 0 }).item<float>();
  float fy = intri.index({ 0, 1, 1 }).item<float>();
  int res_w = 128;
  int res_h = std::round(res_w / half_w * half_h);

  Tensor i = torch::linspace(.5f, half_h * 2.f - .5f, res_h, CUDAFloat);
  Tensor j = torch::linspace(.5f, half_w * 2.f - .5f, res_w, CUDAFloat);
  auto ijs = torch::meshgrid({i, j}, "ij");
  i = ijs[0].reshape({-1});
  j = ijs[1].reshape({-1});
  Tensor cam_coords = torch::stack({ (j - cx) / fx, -(i - cy) / fy, -torch::ones_like(j, CUDAFloat)}, -1); // [ n_pix, 3 ]
  Tensor rays_d = torch::matmul(c2w.index({ Slc(), None, Slc(0, 3), Slc(0, 3) }), cam_coords.index({None, Slc(), Slc(), None})).index({"...", 0});  // [ n_cams, n_pix, 3 ]
  Tensor rays_o = c2w.index({Slc(), None, Slc(0, 3), 3}).repeat({1, res_h * res_w, 1 });
  Tensor a = ((center - bbox_side_len * .5f).index({None, None}) - rays_o) / rays_d;
  Tensor b = ((center + bbox_side_len * .5f).index({None, None}) - rays_o) / rays_d;
  a = torch::nan_to_num(a, 0.f, 1e6f, -1e6f);
  b = torch::nan_to_num(b, 0.f, 1e6f, -1e6f);
  Tensor aa = torch::maximum(a, b);
  Tensor bb = torch::minimum(a, b);
  auto [ far, far_idx ] = torch::min(aa, -1);
  auto [ near, near_idx ] = torch::max(bb, -1);
  far = torch::minimum(far, bound.index({Slc(), None, 1}));
  near = torch::maximum(near, bound.index({Slc(), None, 0}));
  Tensor mask = (far > near).to(torch::kFloat32).sum(-1);
  Tensor good = torch::where(mask > 0)[0].to(torch::kInt32).to(torch::kCPU);
  std::vector<int> ret;
  for (int idx = 0; idx < good.sizes()[0]; idx++) {
    ret.push_back(good[idx].item<int>());
  }
  return ret;
}

}

std::tuple<Tensor, Tensor> PCA(const Tensor& pts) {
  Tensor mean = pts.mean(0, true);
  Tensor moved = pts - mean;
  Tensor cov = torch::matmul(moved.unsqueeze(-1), moved.unsqueeze(1));  // [ n_pts, n_frames, n_frames ];
  cov = cov.mean(0);
  auto [ L, V ] = torch::linalg_eigh(cov);
  L = L.to(torch::kFloat32);
  V = V.to(torch::kFloat32);
  auto [ L_sorted, indices ] = torch::sort(L, 0, true);
  V = V.permute({1, 0}).contiguous().index({ indices }).permute({1, 0}).contiguous();   // { in_dims, 3 }
  L = L.index({ indices }).contiguous();
  return { L, V };
}

// -------------------------------------------- Sampler ------------------------------------------------


PersSampler::PersSampler(GlobalDataPool* global_data_pool) {
  ScopeWatch watch("PersSampler::PersSampler");
  global_data_pool_ = global_data_pool;
  auto config = global_data_pool->config_["pts_sampler"];
  auto dataset = RE_INTER(Dataset*, global_data_pool_->dataset_);

  float split_dist_thres = config["split_dist_thres"].as<float>();
  sub_div_milestones_ = config["sub_div_milestones"].as<std::vector<int>>();
  compact_freq_ = config["compact_freq"].as<int>();
  max_oct_intersect_per_ray_ = config["max_oct_intersect_per_ray"].as<int>();
  std::reverse(sub_div_milestones_.begin(), sub_div_milestones_.end());

  global_near_ = config["near"].as<float>();
  scale_by_dis_ = config["scale_by_dis"].as<bool>();
  int bbox_levels = config["bbox_levels"].as<int>();
  float bbox_side_len = (1 << (bbox_levels - 1));

  sample_l_ = config["sample_l"].as<float>();
  int max_level = config["max_level"].as<int>();
}

std::vector<Tensor> PersSampler::States() {
  std::vector<Tensor> ret;
  Tensor milestones_ts = torch::from_blob(sub_div_milestones_.data(), sub_div_milestones_.size(), CPUInt).to(torch::kCUDA);
  ret.push_back(milestones_ts);

  return ret;
}

int PersSampler::LoadStates(const std::vector<Tensor>& states, int idx) {
  Tensor milestones_ts = states[idx++].clone().to(torch::kCPU).contiguous();

  sub_div_milestones_.resize(milestones_ts.sizes()[0]);
  std::memcpy(sub_div_milestones_.data(), milestones_ts.data_ptr(), milestones_ts.sizes()[0] * sizeof(int));
  PRINT_VAL(sub_div_milestones_);

  return idx;
}

SampleResultFlex PersSampler::GetSamples(
  const Tensor & rays_o_raw, const Tensor & rays_d_raw, const Tensor & bounds_raw)
{
  Tensor rays_o = rays_o_raw.contiguous();
  Tensor rays_d = (rays_d_raw / torch::linalg_norm(rays_d_raw, 2, -1, true)).contiguous();

  int n_rays = rays_o.sizes()[0];
  Tensor bounds =
    torch::stack(
      {torch::full({n_rays}, global_near_, CUDAFloat), torch::full({n_rays}, 1e8f, CUDAFloat)}, -1)
      .contiguous();

  // do ray marching
  constexpr int MAX_SAMPLE_PER_RAY = 1024;
  const int n_all_pts = n_rays * MAX_SAMPLE_PER_RAY;

  Tensor rays_noise;
  if (global_data_pool_->mode_ == RunningMode::VALIDATE) {
    rays_noise = torch::ones({n_all_pts}, CUDAFloat);
  } else {
    rays_noise = ((torch::rand({n_all_pts}, CUDAFloat) - .5f) + 1.f).contiguous();
  }
  rays_noise.mul_(global_data_pool_->ray_march_fineness_);
  rays_noise = rays_noise.view({n_rays, MAX_SAMPLE_PER_RAY}).contiguous();
  Tensor cum_noise = torch::cumsum(rays_noise, 1) * sample_l_;
  Tensor sampled_t = cum_noise.reshape({n_all_pts}).contiguous();

  rays_o = rays_o.view({n_rays, 1, 3}).contiguous();
  rays_d = rays_d.view({n_rays, 1, 3}).contiguous();
  cum_noise = cum_noise.unsqueeze(-1).contiguous();
  Tensor sampled_pts = rays_o + rays_d * cum_noise;

  Tensor sampled_dists = torch::diff(sampled_pts, 1, 1).norm(2, -1).contiguous();
  sampled_dists = torch::cat({torch::zeros({n_rays, 1}, CUDAFloat), sampled_dists}, 1).contiguous();
  sampled_pts = sampled_pts.view({n_all_pts, 3});
  sampled_dists = sampled_dists.view({n_all_pts}).contiguous();

  Tensor pts_idx_start_end = torch::ones({n_rays, 2}, CUDAInt) * MAX_SAMPLE_PER_RAY;
  Tensor pts_num = pts_idx_start_end.index({Slc(), 0});
  Tensor cum_num = torch::cumsum(pts_num, 0);
  pts_idx_start_end.index_put_({Slc(), 0}, cum_num - pts_num);
  pts_idx_start_end.index_put_({Slc(), 1}, cum_num);

  Tensor sampled_dirs =
    rays_d.expand({-1, MAX_SAMPLE_PER_RAY, -1}).reshape({n_all_pts, 3}).contiguous();
  Tensor sampled_anchors = torch::zeros({n_all_pts, 3}, CUDAInt);
  Tensor first_oct_dis = torch::full({n_rays, 1}, 1e9f, CUDAFloat).contiguous();

  return {
    sampled_pts,     sampled_dirs,      sampled_dists, sampled_t,
    sampled_anchors, pts_idx_start_end, first_oct_dis,
  };
}
