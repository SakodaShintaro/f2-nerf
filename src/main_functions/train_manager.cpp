//
// Created by ppwang on 2022/5/6.
//

#include "train_manager.hpp"

#include "../CustomOps/CustomOps.hpp"
#include "../stop_watch.hpp"
#include "../utils.hpp"

#include <experimental/filesystem>
#include <opencv2/core.hpp>

#include <fmt/core.h>

namespace fs = std::experimental::filesystem::v1;
using Tensor = torch::Tensor;

TrainManager::TrainManager(const std::string & conf_path)
{
  fs::path p(conf_path);
  fs::path canonical_path = fs::canonical(p);
  const std::string path = canonical_path.string();
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open " + conf_path);
  }

  fs["base_exp_dir"] >> base_exp_dir_;
  fs::create_directories(base_exp_dir_);

  const cv::FileNode train_config = fs["train"];
  pts_batch_size_ = (int)train_config["pts_batch_size"];
  end_iter_ = (int)train_config["end_iter"];
  vis_freq_ = (int)train_config["vis_freq"];
  report_freq_ = (int)train_config["report_freq"];
  save_freq_ = (int)train_config["save_freq"];
  learning_rate_ = (float)train_config["learning_rate"];
  learning_rate_alpha_ = (float)train_config["learning_rate_alpha"];
  learning_rate_warm_up_end_iter_ = (int)train_config["learning_rate_warm_up_end_iter"];
  var_loss_weight_ = (float)train_config["var_loss_weight"];
  var_loss_start_ = (int)train_config["var_loss_start"];
  var_loss_end_ = (int)train_config["var_loss_end"];

  // Dataset
  dataset_ = std::make_shared<Dataset>(fs["dataset_path"].string());
  dataset_->save_inference_params(base_exp_dir_);

  // Renderer
  const bool use_app_emb = (fs["renderer"]["use_app_emb"].string() == "true");
  renderer_ = std::make_shared<Renderer>(use_app_emb, dataset_->n_images_);
  renderer_->to(torch::kCUDA);

  // Optimizer
  optimizer_ = std::make_shared<torch::optim::Adam>(renderer_->optim_param_groups(learning_rate_));
}

void TrainManager::train()
{
  std::ofstream ofs_log(base_exp_dir_ + "/train_log.txt");

  Timer timer;
  timer.start();

  float psnr_smooth = -1.0;
  updated_ada_params();

  for (; iter_step_ < end_iter_;) {
    constexpr float sampled_pts_per_ray_ = 512.f;
    int cur_batch_size = int(pts_batch_size_ / sampled_pts_per_ray_) >> 4 << 4;
    auto [train_rays, gt_colors, emb_idx] = dataset_->sample_random_rays(cur_batch_size);

    Tensor & rays_o = train_rays.origins;
    Tensor & rays_d = train_rays.dirs;

    auto render_result = renderer_->render(rays_o, rays_d, emb_idx, RunningMode::TRAIN);
    Tensor pred_colors = render_result.colors.index({Slc(0, cur_batch_size)});
    Tensor disparity = render_result.disparity;
    Tensor color_loss = torch::sqrt((pred_colors - gt_colors).square() + 1e-4f).mean();

    Tensor disparity_loss = disparity.square().mean();

    Tensor sampled_weights = render_result.weights;
    Tensor idx_start_end = render_result.idx_start_end;
    Tensor sampled_var = CustomOps::WeightVar(sampled_weights, idx_start_end);
    Tensor var_loss = (sampled_var + 1e-2).sqrt().mean();

    float var_loss_weight = 0.f;
    if (iter_step_ > var_loss_end_) {
      var_loss_weight = var_loss_weight_;
    } else if (iter_step_ > var_loss_start_) {
      var_loss_weight = float(iter_step_ - var_loss_start_) /
                        float(var_loss_end_ - var_loss_start_) * var_loss_weight_;
    }

    Tensor loss = color_loss + var_loss * var_loss_weight;

    float mse = (pred_colors - gt_colors).square().mean().item<float>();
    float psnr = 20.f * std::log10(1 / std::sqrt(mse));
    psnr_smooth = psnr_smooth < 0.f ? psnr : psnr * .1f + psnr_smooth * .9f;
    CHECK(!std::isnan(pred_colors.mean().item<float>()));
    CHECK(!std::isnan(gt_colors.mean().item<float>()));
    CHECK(!std::isnan(mse));

    // There can be some cases that the output colors have no grad due to the occupancy grid.
    if (loss.requires_grad()) {
      optimizer_->zero_grad();
      loss.backward();
      optimizer_->step();
    }

    iter_step_++;

    if (iter_step_ % vis_freq_ == 0) {
      visualize_image(0);
    }

    if (iter_step_ % save_freq_ == 0) {
      fs::remove_all(base_exp_dir_ + "/checkpoints/latest");
      fs::create_directories(base_exp_dir_ + "/checkpoints/latest");
      torch::save(renderer_, base_exp_dir_ + "/checkpoints/latest/renderer.pt");
    }
    const int64_t total_sec = timer.elapsed_seconds();
    const int64_t total_m = total_sec / 60;
    const int64_t total_s = total_sec % 60;

    if (iter_step_ % report_freq_ == 0) {
      std::stringstream ss;
      ss << std::fixed;
      ss << "Time: " << std::setw(2) << std::setfill('0') << total_m << ":" << std::setw(2)
         << std::setfill('0') << total_s << " ";
      ss << "Iter: " << std::setw(6) << iter_step_ << " ";
      ss << "PSNR: " << psnr_smooth << " ";
      ss << "LOSS: " << color_loss.item<float>() << " ";
      ss << "LR: " << optimizer_->param_groups()[0].options().get_lr();
      const std::string log_str = ss.str();
      std::cout << log_str << std::endl;
      ofs_log << log_str << std::endl;
    }
    updated_ada_params();
  }

  std::cout << "Train done" << std::endl;
}

void TrainManager::updated_ada_params()
{
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

void TrainManager::visualize_image(int idx)
{
  torch::NoGradGuard no_grad_guard;

  auto [rays_o, rays_d] = dataset_->get_all_rays_of_camera(idx);

  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors =
    torch::zeros({n_rays, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  Tensor pred_disps =
    torch::zeros({n_rays, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o = rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d = rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result = renderer_->render(cur_rays_o, cur_rays_d, Tensor(), RunningMode::VALIDATE);
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_disps.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
  }
  pred_disps = pred_disps / pred_disps.max();

  int H = dataset_->height_;
  int W = dataset_->width_;

  Tensor img_tensor = torch::cat(
    {dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 3}),
     pred_colors.reshape({H, W, 3}), pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})},
    1);
  fs::create_directories(base_exp_dir_ + "/images");
  std::stringstream ss;
  ss << iter_step_ << "_" << idx << ".png";
  utils::write_image_tensor(base_exp_dir_ + "/images/" + ss.str(), img_tensor);
}
