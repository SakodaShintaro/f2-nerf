//
// Created by ppwang on 2022/10/4.
//

#ifndef SANR_TCNNWP_H
#define SANR_TCNNWP_H

#include <tiny-cuda-nn/cpp_api.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

class TCNNWP {
  using Tensor = torch::Tensor;
public:
  TCNNWP(const YAML::Node & config, int d_in, int d_out, int d_hidden, int n_hidden_layers);

  Tensor Query(const Tensor& pts);

  int d_in_, d_out_, d_hidden_, n_hidden_layers_;
  std::unique_ptr<tcnn::cpp::Module> module_;

  Tensor params_;
  Tensor query_pts_, query_output_;

  tcnn::cpp::Context tcnn_ctx_;

  float loss_scale_ = 128.f;
};

class TCNNWPInfo : public torch::CustomClassHolder {
public:
  TCNNWP* tcnn_ = nullptr;
};

namespace torch::autograd {

class TCNNWPFunction : public Function<TCNNWPFunction> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor input,
                               Tensor params,
                               IValue TCNNWPInfo);

  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}

#endif //SANR_TCNNWP_H
