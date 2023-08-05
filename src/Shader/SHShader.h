//
// Created by ppwang on 2022/10/8.
//

#ifndef SANR_SHSHADER_H
#define SANR_SHSHADER_H

#include "../Field/TCNNWP.h"
#include "../Utils/Pipe.h"

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class SHShader : public Pipe {
  using Tensor = torch::Tensor;
public:
  SHShader(const YAML::Node & config);
  Tensor Query(const Tensor& feats, const Tensor& dirs);
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups(float lr) override;
  int LoadStates(const std::vector<Tensor>& states, int) override;

  Tensor SHEncode(const Tensor& dirs);

  std::unique_ptr<TCNNWP> mlp_;

  int d_in_, d_out_;
  int d_hidden_, n_hiddens_;
  int degree_;
};


#endif //SANR_SHSHADER_H
