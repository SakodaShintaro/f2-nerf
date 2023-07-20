//
// Created by ppwang on 2022/9/26.
//

#include "PersSampler.h"
#include "../Utils/Utils.h"
#define MAX_STACK_SIZE 48
#define MAX_OCT_INTERSECT_PER_RAY 1024
#define MAX_SAMPLE_PER_RAY 1024

#define OCC_WEIGHT_BASE 512
#define ABS_WEIGHT_THRES 0.01
#define REL_WEIGHT_THRES 0.1

#define OCC_ALPHA_BASE 32
#define ABS_ALPHA_THRES 0.02
#define REL_ALPHA_THRES 0.1

using Tensor = torch::Tensor;

void __device__ QueryFrameTransform(const TransInfo& trans,
                                    const Wec3f& cur_xyz,
                                    Wec3f* fill_xyz) {
  Wec4f cur_xyz_ext;
  cur_xyz_ext = cur_xyz.homogeneous();
  Eigen::Matrix<float, N_PROS, 1> transed_vals;
#pragma unroll
  for (int i = 0; i < N_PROS; i++) {
    Wec2f xz = trans.w2xz[i] * cur_xyz_ext;
    transed_vals(i, 0) = xz[0] / xz[1];
  }

  Wec3f weighted = trans.weight * transed_vals;
  *fill_xyz = weighted;
}

__global__ void MarkVistNodeKernel(int n_rays,
                                   int* pts_idx_start_end,
                                   int* oct_indices,
                                   float* sampled_weights,
                                   float* sampled_alpha,
                                   int* visit_weight_adder,
                                   int* visit_alpha_adder,
                                   int* visit_mark,
                                   int* visit_cnt) {
  const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= n_rays) { return; }
  const int pts_idx_start = pts_idx_start_end[ray_idx * 2];
  const int pts_idx_end   = pts_idx_start_end[ray_idx * 2 + 1];
  if (pts_idx_start >= pts_idx_end) { return; }
  float max_weight = 0.f;
  float max_alpha = 0.f;
  for (int pts_idx = pts_idx_start; pts_idx < pts_idx_end; pts_idx++) {
    max_weight = fmaxf(max_weight, sampled_weights[pts_idx]);
    max_alpha = fmaxf(max_alpha, sampled_alpha[pts_idx]);
  }

  const float weight_thres = fminf(max_weight * REL_WEIGHT_THRES, ABS_WEIGHT_THRES);
  const float alpha_thres = fminf(max_alpha * REL_ALPHA_THRES, ABS_ALPHA_THRES);

  float cur_oct_weight = 0.f;
  float cur_oct_alpha = 0.f;
  int cur_oct_idx = -1;
  int cur_visit_cnt = 0;
  for (int pts_idx = pts_idx_start; pts_idx < pts_idx_end; pts_idx++) {
    if (cur_oct_idx != oct_indices[pts_idx]) {
      if (cur_oct_idx >= 0) {
        atomicMax(visit_weight_adder + cur_oct_idx, cur_oct_weight > weight_thres ? OCC_WEIGHT_BASE : -1);
        atomicMax(visit_alpha_adder + cur_oct_idx, cur_oct_alpha > alpha_thres ? OCC_ALPHA_BASE : -1);
        atomicMax(visit_cnt + cur_oct_idx, cur_visit_cnt);
        visit_mark[cur_oct_idx] = 1;
      }
      cur_oct_idx = oct_indices[pts_idx];
      cur_oct_weight = 0.f;
      cur_oct_alpha = 0.f;
      cur_visit_cnt = 0;
    }
    cur_oct_weight = fmaxf(cur_oct_weight, sampled_weights[pts_idx]);
    cur_oct_alpha = fmaxf(cur_oct_alpha, sampled_alpha[pts_idx]);
    cur_visit_cnt += 1;
  }
  if (cur_oct_idx >= 0) {
    atomicMax(visit_weight_adder + cur_oct_idx, cur_oct_weight > weight_thres ? OCC_WEIGHT_BASE : -1);
    atomicMax(visit_alpha_adder + cur_oct_idx, cur_oct_alpha > alpha_thres ? OCC_ALPHA_BASE : -1);
    atomicMax(visit_cnt + cur_oct_idx, cur_visit_cnt);
    visit_mark[cur_oct_idx] = 1;
  }
}

__global__ void MarkInvalidNodes(int n_nodes, int* node_weight_stats, int* node_alpha_stats, TreeNode* nodes) {
  int oct_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (oct_idx >= n_nodes) { return; }
  if (node_weight_stats[oct_idx] < 0 || node_alpha_stats[oct_idx] < 0) {
    nodes[oct_idx].trans_idx = -1;
  }
}

void PersSampler::UpdateOctNodes(const SampleResultFlex& sample_result,
                                 const Tensor& sampled_weight,
                                 const Tensor& sampled_alpha) {
  Tensor oct_indices = sample_result.anchors.index({"...", 1}).contiguous();
  const Tensor& sampled_dists = sample_result.dt.contiguous();
  const Tensor& pts_idx_start_end = sample_result.pts_idx_bounds;
  CK_CONT(sampled_dists);
  CK_CONT(oct_indices);
  CK_CONT(sampled_weight);
  CK_CONT(sampled_alpha);
  CK_CONT(pts_idx_start_end);

  const int n_nodes = pers_octree_->tree_nodes_.size();
  const int n_pts = oct_indices.size(0);
  const int n_rays = pts_idx_start_end.size(0);
  CHECK_EQ(n_pts, sampled_weight.size(0));
  CHECK_EQ(n_pts, sampled_alpha.size(0));
  CHECK_EQ(n_pts, sampled_dists.size(0));

  Tensor visit_weight_adder = torch::full({ n_nodes }, -1, CUDAInt);
  Tensor visit_alpha_adder = torch::full({ n_nodes }, -1,CUDAInt);
  Tensor visit_mark = torch::zeros({ n_nodes }, CUDAInt);
  Tensor& visit_cnt = pers_octree_->tree_visit_cnt_;
  CK_CONT(visit_weight_adder);
  CK_CONT(visit_alpha_adder);
  CK_CONT(visit_mark);
  CK_CONT(visit_cnt);

  {
    dim3 block_dim = LIN_BLOCK_DIM(n_rays);
    dim3 grid_dim  = LIN_GRID_DIM(n_rays);
    MarkVistNodeKernel<<<grid_dim, block_dim>>>(n_rays,
                                                pts_idx_start_end.data_ptr<int>(),
                                                oct_indices.data_ptr<int>(),
                                                sampled_weight.data_ptr<float>(),
                                                sampled_alpha.data_ptr<float>(),
                                                visit_weight_adder.data_ptr<int>(),
                                                visit_alpha_adder.data_ptr<int>(),
                                                visit_mark.data_ptr<int>(),
                                                visit_cnt.data_ptr<int>());

  }

  Tensor& node_weight_stats = pers_octree_->tree_weight_stats_;
  Tensor occ_weight_mask = (visit_weight_adder > 0).to(torch::kInt32);
  node_weight_stats = torch::maximum(node_weight_stats, occ_weight_mask * visit_weight_adder);
  node_weight_stats += (visit_mark * (1 - occ_weight_mask) * visit_weight_adder);
  node_weight_stats.clamp_(-100, 1 << 20);
  node_weight_stats = node_weight_stats.contiguous();
  CK_CONT(node_weight_stats);

  Tensor& node_alpha_stats = pers_octree_->tree_alpha_stats_;
  Tensor occ_alpha_mask = (visit_alpha_adder > 0).to(torch::kInt32);
  node_alpha_stats = torch::maximum(node_alpha_stats, occ_alpha_mask * visit_alpha_adder);
  node_alpha_stats += (visit_mark * (1 - occ_alpha_mask) * visit_alpha_adder);
  node_alpha_stats.clamp_(-100, 1 << 20);
  node_alpha_stats = node_alpha_stats.contiguous();
  CK_CONT(node_alpha_stats);

  {
    dim3 block_dim = LIN_BLOCK_DIM(n_nodes);
    dim3 grid_dim  = LIN_GRID_DIM(n_nodes);
    MarkInvalidNodes<<<grid_dim, block_dim>>>(
        n_nodes,
        node_weight_stats.data_ptr<int>(),
            node_alpha_stats.data_ptr<int>(),
        RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()));
  }

  while (!sub_div_milestones_.empty() && sub_div_milestones_.back() <= global_data_pool_->iter_step_) {
    pers_octree_->ProcOctree(true, true, sub_div_milestones_.back() <= 0);
    pers_octree_->MarkInvisibleNodes();
    pers_octree_->ProcOctree(true, false, false);
    sub_div_milestones_.pop_back();
  }

  if (global_data_pool_->iter_step_ % compact_freq_ == 0) {
    pers_octree_->ProcOctree(true, false, false);
  }
}


__device__ int CheckVisible(const Wec3f& center, float side_len,
                            const Watrix33f& intri, const Watrix34f& w2c, const Wec2f& bound) {
  Wec3f cam_pt = w2c * center.homogeneous();
  float radius = side_len * 0.707;
  if (-cam_pt.z() < bound(0) - radius ||
      -cam_pt.z() > bound(1) + radius) {
    return 0;
  }
  if (cam_pt.norm() < radius) {
    return 1;
  }

  float cx = intri(0, 2);
  float cy = intri(1, 2);
  float fx = intri(0, 0);
  float fy = intri(1, 1);
  float bias_x = radius / -cam_pt.z() * fx;
  float bias_y = radius / -cam_pt.z() * fy;
  float img_pt_x = cam_pt.x() / -cam_pt.z() * fx;
  float img_pt_y = cam_pt.y() / -cam_pt.z() * fy;
  if (img_pt_x + bias_x < -cx || img_pt_x > cx + bias_x ||
      img_pt_y + bias_y < -cy || img_pt_y > cy + bias_y) {
    return 0;
  }
  return 1;
}

__global__ void MarkInvisibleNodesKernel(int n_nodes, int n_cams,
                                         TreeNode* tree_nodes,
                                         Watrix33f* intris, Watrix34f* w2cs, Wec2f* bounds) {
  int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_idx >= n_nodes) { return; }
  int n_visible_cams = 0;
  for (int cam_idx = 0; cam_idx < n_cams; cam_idx++) {
    n_visible_cams += CheckVisible(tree_nodes[node_idx].center,
                                   tree_nodes[node_idx].side_len,
                                   intris[cam_idx],
                                   w2cs[cam_idx],
                                   bounds[cam_idx]);
  }
  if (n_visible_cams < 1) {
    tree_nodes[node_idx].trans_idx = -1;
  }
}

void PersOctree::MarkInvisibleNodes() {
  int n_nodes = tree_nodes_.size();
  int n_cams = intri_.size(0);

  CK_CONT(intri_);
  CK_CONT(w2c_);
  CK_CONT(bound_);

  dim3 block_dim = LIN_BLOCK_DIM(n_nodes);
  dim3 grid_dim = LIN_GRID_DIM(n_nodes);
  MarkInvisibleNodesKernel<<<grid_dim, block_dim>>>(
      n_nodes, n_cams,
      RE_INTER(TreeNode*, tree_nodes_gpu_.data_ptr()),
      RE_INTER(Watrix33f*, intri_.data_ptr()),
      RE_INTER(Watrix34f*, w2c_.data_ptr()),
      RE_INTER(Wec2f*, bound_.data_ptr())
  );
}