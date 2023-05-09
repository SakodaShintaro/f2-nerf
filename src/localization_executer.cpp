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

float LocalizationExecuter::CalcScore(const Tensor pose, Tensor gt_image)
{
  torch::NoGradGuard no_grad_guard;
  auto [rays_o, rays_d, bounds] = dataset_->RaysFromPose(pose);
  auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(rays_o, rays_d, bounds);

  const int H = dataset_->height_;
  const int W = dataset_->width_;

  Tensor pred_img = pred_colors.view({H, W, 3});
  pred_img = pred_img.clip(0.f, 1.f);

  gt_image = gt_image.view({H, W, 3});

  static int cnt = 0;

  std::stringstream ss;
  ss << base_exp_dir_ << "/localization_result/pred_img_";
  ss << std::setfill('0') << std::setw(4) << cnt;
  ss << ".png";
  Utils::WriteImageTensor(ss.str(), pred_img);
  cnt++;

  Tensor diff = pred_img - gt_image;
  Tensor mse = (diff * diff).mean(-1);
  Tensor psnr = 10.f * torch::log10(1.f / mse);
  return psnr.mean().item<float>();
}

void LocalizationExecuter::Localize()
{
  std::cout << "start Localize" << std::endl;
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  const int H = dataset_->height_;
  const int W = dataset_->width_;
  std::cout << "H = " << H << ", W = " << W << std::endl;

  constexpr float noise_std = 0.2f;
  constexpr int NUM_SEARCH = 0;
  fs::create_directories(base_exp_dir_ + "/localization_result");
  std::cout << std::fixed << std::setprecision(2);

  {
    for (int i : dataset_->test_set_) {
      std::string filename = fmt::format("/localization_result/{:08d}.tsv", i);
      std::ofstream ofs(base_exp_dir_ + filename);
      ofs << std::fixed << std::setprecision(1);
      StopWatch stop_watch;
      for (int x = -NUM_SEARCH; x <= NUM_SEARCH; x++) {
        for (int y = -NUM_SEARCH; y <= NUM_SEARCH; y++) {
          Tensor pose = dataset_->poses_[i].clone();
          pose[0][3] += x * noise_std;
          pose[1][3] += y * noise_std;
          float psnr = CalcScore(pose, dataset_->image_tensors_[i]);
          ofs << psnr << (y == NUM_SEARCH ? "\n" : "\t");
        }
      }
      std::cout << "Finish localize " << i << ", time = " << stop_watch.TimeDuration() << " sec"
                << std::endl;
    }
  }

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
