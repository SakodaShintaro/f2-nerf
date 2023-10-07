//
// Created by ppwang on 2022/5/6.
//

#ifndef F2_NERF__TRAIN_MANAGER_HPP_
#define F2_NERF__TRAIN_MANAGER_HPP_

#include "../dataset.hpp"
#include "../renderer.hpp"

#include <torch/torch.h>

#include <memory>
#include <string>
#include <tuple>

class TrainManager
{
  using Tensor = torch::Tensor;

public:
  TrainManager(const std::string & conf_path);

  void train();

  void update_ada_params();

  // data
  std::string base_exp_dir_;

  unsigned iter_step_ = 0;
  unsigned end_iter_;
  unsigned report_freq_, vis_freq_, save_freq_;
  unsigned pts_batch_size_;

  int var_loss_start_, var_loss_end_;
  float learning_rate_, learning_rate_alpha_, learning_rate_warm_up_end_iter_;
  float var_loss_weight_;

  std::shared_ptr<Dataset> dataset_;
  std::shared_ptr<Renderer> renderer_;
  std::shared_ptr<torch::optim::Adam> optimizer_;
};

#endif  // F2_NERF__TRAIN_MANAGER_HPP_
