#ifndef GRID_ENCODER_HPP
#define GRID_ENCODER_HPP

#include "../Utils/Pipe.h"
#include "TCNNWP.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#define N_CHANNELS 2
#define N_LEVELS 16
#define RES_FINE_POW_2 10.f  // 1024
#define RES_BASE_POW_2 3.f   // 8

class GridEncoder : public Pipe
{
  using Tensor = torch::Tensor;

public:
  GridEncoder(const YAML::Node & config);

  Tensor Query(torch::Tensor inputs, double bound = 1);

  int LoadStates(const std::vector<Tensor> & states, int idx) override;
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr) override;

  // Member Variables
  int64_t input_dim, num_levels, level_dim, log2_hashmap_size, base_resolution, output_dim,
    gridtype_id, interp_id, max_params, n_params;
  double per_level_scale, init_std;
  bool align_corners;
  std::string gridtype, interpolation;

  torch::Tensor embeddings, offsets, idx, grid_sizes;

  std::unique_ptr<TCNNWP> mlp_;
};

// class GridEncoderInfo : public torch::CustomClassHolder
// {
// public:
//   GridEncoder * hash3d_ = nullptr;
// };

namespace torch::autograd
{

class GridEncoderFunction : public Function<GridEncoderFunction>
{
public:
  static variable_list forward(
    AutogradContext * ctx, torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
    float per_level_scale, float base_resolution, bool calc_grad_inputs = false,
    int32_t gridtype = 0, bool align_corners = false, int32_t interpolation = 0);
  static variable_list backward(AutogradContext * ctx, variable_list grad_output);
};

}  // namespace torch::autograd

#endif
