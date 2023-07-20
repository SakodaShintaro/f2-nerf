//
// Created by ppwang on 2022/9/26.
//

#ifndef SANR_PERSSAMPLER_H
#define SANR_PERSSAMPLER_H
#include "PtsSampler.h"
#include "Eigen/Eigen"

#define INIT_NODE_STAT 1000
#define N_PROS 12
#define PersMatType Eigen::Matrix<float, 2, 4, Eigen::RowMajor>
#define TransWetType Eigen::Matrix<float, 3, N_PROS, Eigen::RowMajor>

struct alignas(32) TransInfo {
  PersMatType w2xz[N_PROS];
  TransWetType weight;
  Wec3f center;
  float dis_summary;
};

struct alignas(32) TreeNode {
  Wec3f center;
  float side_len;
  int parent;
  int childs[8];
  bool is_leaf_node;
  int trans_idx;
};

struct alignas(32) EdgePool {
  int t_idx_a;
  int t_idx_b;
  Wec3f center;
  Wec3f dir_0;
  Wec3f dir_1;
};

class PersSampler : public PtsSampler {
  using Tensor = torch::Tensor;
public:
  PersSampler(GlobalDataPool* global_data_pool);
  SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds) override;

  std::vector<Tensor> States() override;
  int LoadStates(const std::vector<Tensor>& states, int idx) override;

  std::vector<int> sub_div_milestones_;
  int compact_freq_;
  int max_oct_intersect_per_ray_;
  float global_near_;
  float sample_l_;
  bool scale_by_dis_;
};

#endif //SANR_PERSSAMPLER_H
