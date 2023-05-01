//
// Created by Shintaro Sakoda on 2023/04/29.
//

#include "localization_executer.hpp"

#include "Utils/CustomOps/CustomOps.h"
#include "Utils/StopWatch.h"
#include "Utils/Utils.h"
#include "Utils/cnpy.h"

#include <experimental/filesystem>  // GCC 7.5?

#include <fmt/core.h>

namespace fs = std::experimental::filesystem::v1;
using Tensor = torch::Tensor;

LocalizationExecuter::LocalizationExecuter(const std::string & conf_path)
{
  global_data_pool_ = std::make_unique<GlobalDataPool>(conf_path);
  const auto & config = global_data_pool_->config_;
  case_name_ = config["case_name"].as<std::string>();
  base_dir_ = config["base_dir"].as<std::string>();

  base_exp_dir_ = config["base_exp_dir"].as<std::string>();
  global_data_pool_->base_exp_dir_ = base_exp_dir_;

  fs::create_directories(base_exp_dir_);

  pts_batch_size_ = config["train"]["pts_batch_size"].as<int>();
  end_iter_ = config["train"]["end_iter"].as<int>();
  vis_freq_ = config["train"]["vis_freq"].as<int>();
  report_freq_ = config["train"]["report_freq"].as<int>();
  stats_freq_ = config["train"]["stats_freq"].as<int>();
  save_freq_ = config["train"]["save_freq"].as<int>();
  learning_rate_ = config["train"]["learning_rate"].as<float>();
  learning_rate_alpha_ = config["train"]["learning_rate_alpha"].as<float>();
  learning_rate_warm_up_end_iter_ = config["train"]["learning_rate_warm_up_end_iter"].as<int>();
  ray_march_init_fineness_ = config["train"]["ray_march_init_fineness"].as<float>();
  ray_march_fineness_decay_end_iter_ =
    config["train"]["ray_march_fineness_decay_end_iter"].as<int>();
  tv_loss_weight_ = config["train"]["tv_loss_weight"].as<float>();
  disp_loss_weight_ = config["train"]["disp_loss_weight"].as<float>();
  var_loss_weight_ = config["train"]["var_loss_weight"].as<float>();
  var_loss_start_ = config["train"]["var_loss_start"].as<int>();
  var_loss_end_ = config["train"]["var_loss_end"].as<int>();

  // Dataset
  dataset_ = std::make_unique<Dataset>(global_data_pool_.get());

  // Renderer
  renderer_ = std::make_unique<Renderer>(global_data_pool_.get(), dataset_->n_images_);

  // Optimizer
  optimizer_ = std::make_unique<torch::optim::Adam>(renderer_->OptimParamGroups());

  if (config["is_continue"].as<bool>()) {
    LoadCheckpoint(base_exp_dir_ + "/checkpoints/latest");
  } else {
    std::cerr << "is_continue must be true!" << std::endl;
    std::exit(1);
  }

  if (config["reset"] && config["reset"].as<bool>()) {
    renderer_->Reset();
  }
}

void LocalizationExecuter::LoadCheckpoint(const std::string & path)
{
  {
    Tensor scalars;
    torch::load(scalars, path + "/scalars.pt");
    iter_step_ = std::round(scalars[0].item<float>());
    UpdateAdaParams();
  }

  {
    std::vector<Tensor> scene_states;
    torch::load(scene_states, path + "/renderer.pt");
    renderer_->LoadStates(scene_states, 0);
  }
}

void LocalizationExecuter::UpdateAdaParams()
{
  // Update ray march fineness
  if (iter_step_ >= ray_march_fineness_decay_end_iter_) {
    global_data_pool_->ray_march_fineness_ = 1.f;
  } else {
    float progress = float(iter_step_) / float(ray_march_fineness_decay_end_iter_);
    global_data_pool_->ray_march_fineness_ =
      std::exp(std::log(1.f) * progress + std::log(ray_march_init_fineness_) * (1.f - progress));
  }
  // Update learning rate
  float lr_factor;
  if (iter_step_ >= learning_rate_warm_up_end_iter_) {
    float progress = float(iter_step_ - learning_rate_warm_up_end_iter_) /
                     float(end_iter_ - learning_rate_warm_up_end_iter_);
    lr_factor = (1.f - learning_rate_alpha_) * (std::cos(progress * float(M_PI)) * .5f + .5f) +
                learning_rate_alpha_;
  } else {
    lr_factor = float(iter_step_) / float(learning_rate_warm_up_end_iter_);
  }
  float lr = learning_rate_ * lr_factor;
  for (auto & g : optimizer_->param_groups()) {
    g.options().set_lr(lr);
  }
}

std::tuple<Tensor, Tensor, Tensor> LocalizationExecuter::RenderWholeImage(
  Tensor rays_o, Tensor rays_d, Tensor bounds)
{
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor first_oct_disp = torch::full({n_rays, 1}, 1.f, CPUFloat);
  Tensor pred_disp = torch::zeros({n_rays, 1}, CPUFloat);

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o = rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d = rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_bounds = bounds.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result = renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor());
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
    if (!render_result.first_oct_dis.sizes().empty()) {
      Tensor & ret_first_oct_dis = render_result.first_oct_dis;
      if (ret_first_oct_dis.has_storage()) {
        Tensor cur_first_oct_dis = render_result.first_oct_dis.detach().to(torch::kCPU);
        first_oct_disp.index_put_({Slc(i, i_high)}, cur_first_oct_dis);
      }
    }
  }
  pred_disp = pred_disp / pred_disp.max();
  first_oct_disp = first_oct_disp.min() / first_oct_disp;

  return {pred_colors, first_oct_disp, pred_disp};
}

void LocalizationExecuter::VisualizeImage(int idx)
{
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  auto [rays_o, rays_d, bounds] = dataset_->RaysOfCamera(idx);
  auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(rays_o, rays_d, bounds);

  int H = dataset_->height_;
  int W = dataset_->width_;

  Tensor img_tensor = torch::cat(
    {dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 3}),
     pred_colors.reshape({H, W, 3}), first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
     pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})},
    1);
  fs::create_directories(base_exp_dir_ + "/images");
  Utils::WriteImageTensor(
    base_exp_dir_ + "/images/" + fmt::format("{}_{}.png", iter_step_, idx), img_tensor);

  global_data_pool_->mode_ = prev_mode;
}

void LocalizationExecuter::Localize()
{
  std::cout << "start Localize" << std::endl;
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  float psnr_sum = 0.f;
  float cnt = 0.f;
  YAML::Node out_info;
  {
    for (int i : dataset_->test_set_) {
      std::cout << "i = " << i << std::endl;
      auto [rays_o, rays_d, bounds] = dataset_->RaysOfCamera(i);
      auto [pred_colors, first_oct_dis, pred_disps] =
        RenderWholeImage(rays_o, rays_d, bounds);  // At this stage, the returned number is

      int H = dataset_->height_;
      int W = dataset_->width_;

      auto quantify = [](const Tensor & x) {
        return (x.clip(0.f, 1.f) * 255.f).to(torch::kUInt8).to(torch::kFloat32) / 255.f;
      };
      pred_disps = pred_disps.reshape({H, W, 1});
      first_oct_dis = first_oct_dis.reshape({H, W, 1});
      pred_colors = pred_colors.reshape({H, W, 3});
      pred_colors = quantify(pred_colors);
      float mse = (pred_colors.reshape({H, W, 3}) -
                   dataset_->image_tensors_[i].to(torch::kCPU).reshape({H, W, 3}))
                    .square()
                    .mean()
                    .item<float>();
      float psnr = 20.f * std::log10(1 / std::sqrt(mse));
      out_info[fmt::format("{}", i)] = psnr;
      std::cout << fmt::format("{}: {}", i, psnr) << std::endl;
      psnr_sum += psnr;
      cnt += 1.f;
    }
  }
  float mean_psnr = psnr_sum / cnt;
  std::cout << fmt::format("Mean psnr: {}", mean_psnr) << std::endl;
  out_info["mean_psnr"] = mean_psnr;

  std::ofstream info_fout(base_exp_dir_ + "/localization/info.yaml");
  info_fout << out_info;

  global_data_pool_->mode_ = prev_mode;
}

void LocalizationExecuter::Execute()
{
  std::string mode = global_data_pool_->config_["mode"].as<std::string>();
  if (mode == "localize") {
    Localize();
  } else {
    std::cout << "Unknown mode: " << mode << std::endl;
    exit(-1);
  }
}
