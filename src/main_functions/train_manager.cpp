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
  disp_loss_weight_ = (float)train_config["disp_loss_weight"];
  var_loss_weight_ = (float)train_config["var_loss_weight"];
  var_loss_start_ = (int)train_config["var_loss_start"];
  var_loss_end_ = (int)train_config["var_loss_end"];

  std::string data_path;
  fs["dataset_path"] >> data_path;

  // Dataset
  dataset_ = std::make_shared<Dataset>(data_path);
  dataset_->save_inference_params(base_exp_dir_);

  // Renderer
  std::string use_app_emb_str;
  fs["renderer"]["use_app_emb"] >> use_app_emb_str;
  const bool use_app_emb = (use_app_emb_str == "true");
  renderer_ = std::make_shared<Renderer>(use_app_emb, dataset_->n_images_);
  renderer_->to(torch::kCUDA);

  // Optimizer
  optimizer_ = std::make_shared<torch::optim::Adam>(renderer_->OptimParamGroups(learning_rate_));

  std::string is_continue_str;
  fs["is_continue"] >> is_continue_str;
  const bool is_continue = (is_continue_str == "true");
  if (is_continue) {
    LoadCheckpoint(base_exp_dir_ + "/checkpoints/latest");
  }
}

void TrainManager::Train()
{
  std::ofstream ofs_log(base_exp_dir_ + "/train_log.txt");

  StopWatch clock;
  Timer timer;
  timer.start();

  float psnr_smooth = -1.0;
  UpdateAdaParams();

  for (; iter_step_ < end_iter_;) {
    constexpr float sampled_pts_per_ray_ = 512.f;
    int cur_batch_size = int(pts_batch_size_ / sampled_pts_per_ray_) >> 4 << 4;
    auto [train_rays, gt_colors, emb_idx] = dataset_->sample_random_rays(cur_batch_size);

    Tensor & rays_o = train_rays.origins;
    Tensor & rays_d = train_rays.dirs;
    Tensor & bounds = train_rays.bounds;

    auto render_result = renderer_->Render(rays_o, rays_d, emb_idx, RunningMode::TRAIN);
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

    Tensor loss = color_loss + var_loss * var_loss_weight + disparity_loss * disp_loss_weight_;

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
      VisualizeImage(0);
    }

    if (iter_step_ % save_freq_ == 0) {
      SaveCheckpoint();
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
    UpdateAdaParams();
  }

  std::cout << "Train done" << std::endl;
}

void TrainManager::LoadCheckpoint(const std::string & path)
{
  {
    Tensor scalars;
    torch::load(scalars, path + "/scalars.pt");
    iter_step_ = std::round(scalars[0].item<float>());
    UpdateAdaParams();
  }

  torch::load(renderer_, path + "/renderer.pt");
}

void TrainManager::SaveCheckpoint()
{
  std::stringstream ss;
  ss << base_exp_dir_ << "/checkpoints/" << std::setw(8) << std::setfill('0') << iter_step_;
  std::string output_dir = ss.str();
  fs::create_directories(output_dir);

  fs::remove_all(base_exp_dir_ + "/checkpoints/latest");
  fs::create_directory(base_exp_dir_ + "/checkpoints/latest");
  // scene
  torch::save(renderer_, output_dir + "/renderer.pt");
  fs::create_symlink(
    output_dir + "/renderer.pt", base_exp_dir_ + "/checkpoints/latest/renderer.pt");
  // optimizer
  // torch::save(*(optimizer_), output_dir + "/optimizer.pt");
  // other scalars
  Tensor scalars =
    torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  scalars.index_put_({0}, float(iter_step_));
  torch::save(scalars, output_dir + "/scalars.pt");
  fs::create_symlink(output_dir + "/scalars.pt", base_exp_dir_ + "/checkpoints/latest/scalars.pt");
}

void TrainManager::UpdateAdaParams()
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

std::tuple<Tensor, Tensor> TrainManager::RenderWholeImage(
  Tensor rays_o, Tensor rays_d, Tensor bounds, RunningMode mode)
{
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors =
    torch::zeros({n_rays, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  Tensor pred_disp =
    torch::zeros({n_rays, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o = rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d = rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_bounds = bounds.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result = renderer_->Render(cur_rays_o, cur_rays_d, Tensor(), mode);
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
  }
  pred_disp = pred_disp / pred_disp.max();

  return {pred_colors, pred_disp};
}

void TrainManager::VisualizeImage(int idx)
{
  torch::NoGradGuard no_grad_guard;

  auto [rays_o, rays_d, bounds] = dataset_->get_all_rays_of_camera(idx);
  auto [pred_colors, pred_disps] = RenderWholeImage(rays_o, rays_d, bounds, RunningMode::VALIDATE);

  int H = dataset_->height_;
  int W = dataset_->width_;

  Tensor img_tensor = torch::cat(
    {dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 3}),
     pred_colors.reshape({H, W, 3}), pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})},
    1);
  fs::create_directories(base_exp_dir_ + "/images");
  std::stringstream ss;
  ss << iter_step_ << "_" << idx << ".png";
  Utils::WriteImageTensor(base_exp_dir_ + "/images/" + ss.str(), img_tensor);
}
