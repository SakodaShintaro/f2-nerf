//
// Created by ppwang on 2022/5/6.
//
#pragma once
#include <string>
#include <memory>
#include <tuple>
#include <yaml-cpp/yaml.h>
#include <torch/torch.h>
#include "Dataset/Dataset.h"
#include "Renderer/Renderer.h"

class ExpRunner {
  using Tensor = torch::Tensor;
public:
  ExpRunner(const std::string& conf_path);

  void Execute();
  void Train();
  void RenderAllImages();

  void LoadCheckpoint(const std::string& path);
  void SaveCheckpoint();
  void UpdateAdaParams();
  std::tuple<Tensor, Tensor> RenderWholeImage(
    Tensor rays_o, Tensor rays_d, Tensor bounds, RunningMode mode);
  void VisualizeImage(int idx);

  // data
  std::string case_name_, base_dir_, base_exp_dir_;

  unsigned iter_step_ = 0;
  unsigned end_iter_;
  unsigned report_freq_, vis_freq_, save_freq_;
  unsigned pts_batch_size_;

  float ray_march_init_fineness_;
  int ray_march_fineness_decay_end_iter_;
  int var_loss_start_, var_loss_end_;
  float learning_rate_, learning_rate_alpha_, learning_rate_warm_up_end_iter_;
  float gradient_door_end_iter_;
  float var_loss_weight_, tv_loss_weight_, disp_loss_weight_;

  YAML::Node config_;

  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<Renderer> renderer_;
  std::unique_ptr<torch::optim::Adam> optimizer_;
};
