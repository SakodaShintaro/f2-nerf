#ifndef GRID_ENCODER_HPP
#define GRID_ENCODER_HPP

#include "../Utils/Pipe.h"
#include "TCNNWP.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class GridEncoder : public Pipe
{
  using Tensor = torch::Tensor;

public:
  GridEncoder(const YAML::Node & config);

  Tensor Query(torch::Tensor inputs);

  int LoadStates(const std::vector<Tensor> & states, int idx) override;
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr) override;

  // Member Variables
  torch::Tensor embeddings_, offsets_;

  std::unique_ptr<TCNNWP> mlp_;
};

namespace torch::autograd
{

class GridEncoderFunction : public Function<GridEncoderFunction>
{
public:
  static variable_list forward(
    AutogradContext * ctx, torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
    bool calc_grad_inputs = false);
  static variable_list backward(AutogradContext * ctx, variable_list grad_output);
};

}  // namespace torch::autograd

#endif
