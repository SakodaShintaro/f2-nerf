//
// Created by ppwang on 2022/5/6.
//

#include "ExpRunner.h"
#include <experimental/filesystem>  // GCC 7.5?
#include <fmt/core.h>
#include "Utils/Utils.h"
#include "Utils/cnpy.h"
#include "Utils/StopWatch.h"
#include "Utils/CustomOps/CustomOps.h"

namespace fs = std::experimental::filesystem::v1;
using Tensor = torch::Tensor;

constexpr float kPoseLrCoeff = 0.01;

ExpRunner::ExpRunner(const std::string& conf_path) {
  const YAML::Node & config = YAML::LoadFile(conf_path);
  config_ = config;
  case_name_ = config["case_name"].as<std::string>();
  base_dir_ = config["base_dir"].as<std::string>();

  base_exp_dir_ = config["base_exp_dir"].as<std::string>();

  fs::create_directories(base_exp_dir_);

  pts_batch_size_ = config["train"]["pts_batch_size"].as<int>();
  end_iter_ = config["train"]["end_iter"].as<int>();
  vis_freq_ = config["train"]["vis_freq"].as<int>();
  report_freq_ = config["train"]["report_freq"].as<int>();
  save_freq_ = config["train"]["save_freq"].as<int>();
  learning_rate_ = config["train"]["learning_rate"].as<float>();
  learning_rate_alpha_ = config["train"]["learning_rate_alpha"].as<float>();
  learning_rate_warm_up_end_iter_ = config["train"]["learning_rate_warm_up_end_iter"].as<int>();
  ray_march_init_fineness_ = config["train"]["ray_march_init_fineness"].as<float>();
  ray_march_fineness_decay_end_iter_ = config["train"]["ray_march_fineness_decay_end_iter"].as<int>();
  tv_loss_weight_ = config["train"]["tv_loss_weight"].as<float>();
  disp_loss_weight_ = config["train"]["disp_loss_weight"].as<float>();
  var_loss_weight_ = config["train"]["var_loss_weight"].as<float>();
  var_loss_start_ = config["train"]["var_loss_start"].as<int>();
  var_loss_end_ = config["train"]["var_loss_end"].as<int>();

  // Dataset
  dataset_ = std::make_unique<Dataset>(config);
  dataset_->SaveInferenceParams();

  // Renderer
  renderer_ = std::make_unique<Renderer>(config, dataset_->n_images_);

  std::vector<torch::optim::OptimizerParamGroup> param_groups =
    renderer_->OptimParamGroups(learning_rate_);

  // add learnable_pos_, ori_ to param groups
  std::unique_ptr<torch::optim::AdamOptions> opt =
    std::make_unique<torch::optim::AdamOptions>(learning_rate_ * kPoseLrCoeff);
  opt->betas() = {0.9, 0.99};
  opt->eps() = 1e-15;
  opt->weight_decay() = 1e5;
  std::vector<Tensor> params;
  params.push_back(dataset_->learnable_pos_);
  params.push_back(dataset_->learnable_ori_);
  torch::optim::OptimizerParamGroup pose_delta_group(std::move(params), std::move(opt));

  // Add pose_delta_group to param_groups
  param_groups.push_back(pose_delta_group);

  // Optimizer
  optimizer_ = std::make_unique<torch::optim::Adam>(param_groups);

  if (config["is_continue"].as<bool>()) {
    LoadCheckpoint(base_exp_dir_ + "/checkpoints/latest");
  }
}

void ExpRunner::Train() {
  std::ofstream ofs_log(base_exp_dir_ + "/train_log.txt");

  StopWatch clock;
  Timer timer;
  timer.start();

  float psnr_smooth = -1.0;
  UpdateAdaParams();

  for (; iter_step_ < end_iter_;) {
    constexpr float sampled_pts_per_ray_ = 512.f;
    int cur_batch_size = int(pts_batch_size_ / sampled_pts_per_ray_) >> 4 << 4;
    auto [train_rays, gt_colors, emb_idx] = dataset_->RandRaysData(cur_batch_size, DATA_TRAIN_SET);

    Tensor& rays_o = train_rays.origins;
    Tensor& rays_d = train_rays.dirs;
    Tensor& bounds = train_rays.bounds;

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
    }
    else if (iter_step_ > var_loss_start_) {
      var_loss_weight = float(iter_step_ - var_loss_start_) / float(var_loss_end_ - var_loss_start_) * var_loss_weight_;
    }

    Tensor loss = color_loss + var_loss * var_loss_weight +
                  disparity_loss * disp_loss_weight_;

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
      int t = iter_step_ / vis_freq_;
      int vis_idx;
      vis_idx = (iter_step_ / vis_freq_) % dataset_->test_set_.size();
      vis_idx = dataset_->test_set_[vis_idx];
      VisualizeImage(vis_idx);
    }

    if (iter_step_ % save_freq_ == 0) {
      SaveCheckpoint();
    }
    const int64_t total_sec = timer.elapsed_seconds();
    const int64_t total_m = total_sec / 60;
    const int64_t total_s = total_sec % 60;

    if (iter_step_ % report_freq_ == 0) {
      const std::string log_str = fmt::format(
          "Time: {:>02d}:{:>02d}  Iter: {:>6d}  PSNR: {:.2f}  LOSS: {:.4f}  LR: {:.4f}",
          total_m,
          total_s,
          iter_step_,
          psnr_smooth,
          color_loss.item<float>(),
          optimizer_->param_groups()[0].options().get_lr());
      std::cout << log_str << std::endl;
      ofs_log << log_str << std::endl;
    }
    UpdateAdaParams();
  }
  std::cout << "save pose_delta_" << std::endl;
  // Save dataset_->pose_delta_
  std::ofstream ofs_pose_delta(base_exp_dir_ + "/pose_delta.txt");
  ofs_pose_delta << std::fixed;
  ofs_pose_delta << "x\ty\tz\tax\tay\taz" << std::endl;
  for (int i = 0; i < dataset_->n_images_; i++) {
    ofs_pose_delta << dataset_->learnable_pos_[i][0].item<float>() << "\t";
    ofs_pose_delta << dataset_->learnable_pos_[i][1].item<float>() << "\t";
    ofs_pose_delta << dataset_->learnable_pos_[i][2].item<float>() << "\t";
    ofs_pose_delta << dataset_->learnable_ori_[i][0].item<float>() << "\t";
    ofs_pose_delta << dataset_->learnable_ori_[i][1].item<float>() << "\t";
    ofs_pose_delta << dataset_->learnable_ori_[i][2].item<float>() << std::endl;
  }

  std::cout << "Train done" << std::endl;
}

void ExpRunner::LoadCheckpoint(const std::string& path) {
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

void ExpRunner::SaveCheckpoint() {
  std::string output_dir = base_exp_dir_ + fmt::format("/checkpoints/{:0>8d}", iter_step_);
  fs::create_directories(output_dir);

  fs::remove_all(base_exp_dir_ + "/checkpoints/latest");
  fs::create_directory(base_exp_dir_ + "/checkpoints/latest");
  // scene
  torch::save(renderer_->States(), output_dir + "/renderer.pt");
  fs::create_symlink(output_dir + "/renderer.pt", base_exp_dir_ + "/checkpoints/latest/renderer.pt");
  // optimizer
  // torch::save(*(optimizer_), output_dir + "/optimizer.pt");
  // other scalars
  Tensor scalars = torch::empty({1}, CPUFloat);
  scalars.index_put_({0}, float(iter_step_));
  torch::save(scalars, output_dir + "/scalars.pt");
  fs::create_symlink(output_dir + "/scalars.pt", base_exp_dir_ + "/checkpoints/latest/scalars.pt");
}

void ExpRunner::UpdateAdaParams() {
  // Update ray march fineness
  if (iter_step_ >= ray_march_fineness_decay_end_iter_) {
    renderer_->pts_sampler_->ray_march_fineness_ = 1.f;
  }
  else {
    float progress = float(iter_step_) / float(ray_march_fineness_decay_end_iter_);
    renderer_->pts_sampler_->ray_march_fineness_ =
      std::exp(std::log(1.f) * progress + std::log(ray_march_init_fineness_) * (1.f - progress));
  }
  // Update learning rate
  float lr_factor;
  if (iter_step_ >= learning_rate_warm_up_end_iter_) {
    float progress = float(iter_step_ - learning_rate_warm_up_end_iter_) /
                     float(end_iter_ - learning_rate_warm_up_end_iter_);
    lr_factor = (1.f - learning_rate_alpha_) * (std::cos(progress * float(M_PI)) * .5f + .5f) + learning_rate_alpha_;
  }
  else {
    lr_factor = float(iter_step_) / float(learning_rate_warm_up_end_iter_);
  }
  float lr = learning_rate_ * lr_factor;
  for (auto& g : optimizer_->param_groups()) {
    g.options().set_lr(lr);
  }

  optimizer_->param_groups().back().options().set_lr(lr * kPoseLrCoeff);
  if (iter_step_ == 2500) {
    // pose_delta_groupはparam_groupsの最後に追加されたので、param_groupsの最後の要素としてアクセスする
    std::cout << "Weight Decay to 1e-1" << std::endl;
    auto & options =
      static_cast<torch::optim::AdamOptions &>(optimizer_->param_groups().back().options());
    options.weight_decay(1e-1);
  }
}

std::tuple<Tensor,  Tensor> ExpRunner::RenderWholeImage(Tensor rays_o, Tensor rays_d, Tensor bounds, RunningMode mode) {
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor pred_disp = torch::zeros({n_rays, 1}, CPUFloat);

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

  return { pred_colors, pred_disp };
}

void ExpRunner::VisualizeImage(int idx) {
  torch::NoGradGuard no_grad_guard;

  auto [ rays_o, rays_d, bounds ] = dataset_->RaysOfCamera(idx);
  auto [ pred_colors, pred_disps ] = RenderWholeImage(rays_o, rays_d, bounds, RunningMode::VALIDATE);

  int H = dataset_->height_;
  int W = dataset_->width_;

  Tensor img_tensor = torch::cat({dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 3}),
                                  pred_colors.reshape({H, W, 3}),
                                  pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})}, 1);
  fs::create_directories(base_exp_dir_ + "/images");
  Utils::WriteImageTensor(base_exp_dir_ + "/images/" + fmt::format("{}_{}.png", iter_step_, idx), img_tensor);
}

void ExpRunner::Execute() {
  Train();
}
