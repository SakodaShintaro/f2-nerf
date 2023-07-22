//
// Created by ppwang on 2022/6/20.
//

#pragma once
#include "../Common.h"
#include "../Utils/GlobalDataPool.h"
#include "../Utils/Pipe.h"
#include "Eigen/Eigen"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <memory>

#define INIT_NODE_STAT 1000
#define N_PROS 12
#define PersMatType Eigen::Matrix<float, 2, 4, Eigen::RowMajor>
#define TransWetType Eigen::Matrix<float, 3, N_PROS, Eigen::RowMajor>

struct SampleResultFlex
{
  using Tensor = torch::Tensor;
  Tensor pts;             // [ n_all_pts, 3 ]
  Tensor dirs;            // [ n_all_pts, 3 ]
  Tensor dt;              // [ n_all_pts, 1 ]
  Tensor t;               // [ n_all_pts, 1 ]
  Tensor anchors;         // [ n_all_pts, 3 ]
  Tensor pts_idx_bounds;  // [ n_rays, 2 ] // start, end
  Tensor first_oct_dis;   // [ n_rays, 1 ]
};

struct alignas(32) TransInfo
{
  PersMatType w2xz[N_PROS];
  TransWetType weight;
  Wec3f center;
  float dis_summary;
};

struct alignas(32) TreeNode
{
  Wec3f center;
  float side_len;
  int parent;
  int childs[8];
  bool is_leaf_node;
  int trans_idx;
};

struct alignas(32) EdgePool
{
  int t_idx_a;
  int t_idx_b;
  Wec3f center;
  Wec3f dir_0;
  Wec3f dir_1;
};

class PtsSampler : public Pipe
{
  using Tensor = torch::Tensor;

public:
  PtsSampler(GlobalDataPool * global_data_pool);
  SampleResultFlex GetSamples(const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds);
  GlobalDataPool * global_data_pool_ = nullptr;

  int compact_freq_;
  int max_oct_intersect_per_ray_;
  float global_near_;
  float sample_l_;
  bool scale_by_dis_;
};
