#include "localizer_core.hpp"

#include "../../src/Dataset/Dataset.h"
#include "timer.hpp"

using Tensor = torch::Tensor;

LocalizerCore::LocalizerCore(const std::string & conf_path)
{
  global_data_pool_ = std::make_unique<GlobalDataPool>(conf_path);
  global_data_pool_->mode_ = RunningMode::VALIDATE;
  dataset_ = std::make_unique<Dataset>(global_data_pool_.get());
  renderer_ = std::make_unique<Renderer>(global_data_pool_.get(), dataset_->n_images_);

  const auto & config = global_data_pool_->config_;
  const std::string base_exp_dir = config["base_exp_dir"].as<std::string>();
  std::cout << "base_exp_dir: " << base_exp_dir << std::endl;
  global_data_pool_->base_exp_dir_ = base_exp_dir;
  load_checkpoint(base_exp_dir + "/checkpoints/latest");

  // set
  const float factor = global_data_pool_->config_["dataset"]["factor_to_infer"].as<float>();
  H = 1280 / factor;
  W = 1920 / factor;
  std::cout << "H = " << H << ", W = " << W << ", factor = " << factor << std::endl;
  dataset_->intri_ /= factor;
}

std::tuple<float, Tensor, Tensor> LocalizerCore::monte_carlo_localize(
  Tensor initial_pose, Tensor image_tensor)
{
  torch::NoGradGuard no_grad_guard;

  constexpr float noise_std = 0.2f;
  constexpr int NUM_SEARCH = 0;

  float best_score = -1.0f;
  Tensor best_pose = initial_pose.clone();
  Tensor best_image = image_tensor.clone();

  for (int x = -NUM_SEARCH; x <= NUM_SEARCH; x++) {
    for (int y = -NUM_SEARCH; y <= NUM_SEARCH; y++) {
      Tensor curr_pose = initial_pose.clone();
      curr_pose[0][3] += x * noise_std;
      curr_pose[1][3] += y * noise_std;
      const auto [psnr, pred_image] = calc_score(curr_pose, image_tensor);
      if (psnr > best_score) {
        best_score = psnr;
        best_pose = curr_pose.clone();
        best_image = pred_image.clone();
      }
    }
  }
  return {best_score, best_pose, best_image};
}

std::vector<Particle> LocalizerCore::mc(Tensor initial_pose, Tensor image_tensor)
{
  torch::NoGradGuard no_grad_guard;

  constexpr float noise_std = 0.2f;
  constexpr int NUM_SEARCH = 1;

  std::vector<Tensor> poses;
  for (int x = -NUM_SEARCH; x <= NUM_SEARCH; x++) {
    for (int y = -NUM_SEARCH; y <= NUM_SEARCH; y++) {
      Tensor curr_pose = initial_pose.clone();
      curr_pose[0][3] += x * noise_std;
      curr_pose[1][3] += y * noise_std;
      poses.push_back(curr_pose);
    }
  }

  const std::vector<float> scores = evaluate_poses(poses, image_tensor);
  const int pose_num = poses.size();

  std::vector<Particle> result;
  for (int i = 0; i < pose_num; i++) {
    result.push_back({poses[i], scores[i]});
  }
  return result;
}

void LocalizerCore::load_checkpoint(const std::string & checkpoint_path)
{
  {
    Tensor scalars;
    torch::load(scalars, checkpoint_path + "/scalars.pt");
  }

  {
    std::vector<Tensor> scene_states;
    torch::load(scene_states, checkpoint_path + "/renderer.pt");
    renderer_->LoadStates(scene_states, 0);
  }
}

std::tuple<Tensor, Tensor, Tensor> LocalizerCore::render_all_rays(
  const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds)
{
  torch::NoGradGuard no_grad_guard;
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor first_oct_disp = torch::full({n_rays, 1}, 1.f, CPUFloat);
  Tensor pred_disp = torch::zeros({n_rays, 1}, CPUFloat);

  const int ray_batch_size = H * W;
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

std::tuple<float, Tensor> LocalizerCore::calc_score(const Tensor & pose, const Tensor & image)
{
  torch::NoGradGuard no_grad_guard;
  Timer timer;
  auto [rays_o, rays_d, bounds] = dataset_->RaysFromPose(pose);
  std::cout << "RaysFromPose(): " << timer << std::endl;
  timer.reset();
  auto [pred_colors, first_oct_dis, pred_disps] = render_all_rays(rays_o, rays_d, bounds);
  std::cout << "render_all_rays(): " << timer << std::endl;

  Tensor pred_img = pred_colors.view({H, W, 3});
  pred_img = pred_img.clip(0.f, 1.f);
  pred_img = pred_img.to(image.device());

  Tensor diff = pred_img - image.view({H, W, 3});
  Tensor mse = (diff * diff).mean(-1);
  Tensor psnr = 10.f * torch::log10(1.f / mse);
  return {psnr.mean().item<float>(), pred_img};
}

Tensor LocalizerCore::normalize_position(Tensor pose)
{
  Tensor cam_pos = pose.index({Slc(0, 3), 3}).clone();
  cam_pos = (cam_pos - dataset_->center_.unsqueeze(0)) / dataset_->radius_;
  pose.index_put_({Slc(0, 3), 3}, cam_pos);
  return pose;
}

Tensor LocalizerCore::inverse_normalize_position(Tensor pose)
{
  Tensor cam_pos = pose.index({Slc(0, 3), 3}).clone();
  cam_pos = cam_pos * dataset_->radius_ + dataset_->center_.unsqueeze(0);
  pose.index_put_({Slc(0, 3), 3}, cam_pos);
  return pose;
}

std::vector<float> LocalizerCore::evaluate_poses(
  const std::vector<Tensor> & poses, const Tensor & image)
{
  torch::NoGradGuard no_grad_guard;
  Timer timer;

  const int H = dataset_->height_;
  const int W = dataset_->width_;
  const int batch_size = 32;
  const Tensor i = torch::randint(0, H, batch_size, CUDALong);
  const Tensor j = torch::randint(0, W, batch_size, CUDALong);
  const Tensor ij = torch::stack({i, j}, -1).to(torch::kFloat32);
  std::vector<Tensor> rays_o_vec;
  std::vector<Tensor> rays_d_vec;
  for (const Tensor & pose : poses) {
    auto [rays_o, rays_d] =
      Dataset::Img2WorldRay(pose, dataset_->intri_[0], dataset_->dist_params_[0], ij);
    rays_o_vec.push_back(rays_o);
    rays_d_vec.push_back(rays_d);
  }
  const float near = dataset_->bounds_.index({Slc(), 0}).min().item<float>();
  const float far = dataset_->bounds_.index({Slc(), 1}).max().item<float>();

  const int64_t pose_num = poses.size();
  const int64_t numel = batch_size * pose_num;

  Tensor rays_o = torch::cat(rays_o_vec);  // (numel, 3)
  Tensor rays_d = torch::cat(rays_d_vec);  // (numel, 3)
  Tensor bounds =
    torch::stack({torch::full({numel}, near, CUDAFloat), torch::full({numel}, far, CUDAFloat)}, -1)
      .contiguous();  // (numel, 2)

  std::cout << "RaysFromPose(): " << timer << std::endl;
  timer.reset();
  auto [pred_colors, first_oct_dis, pred_disps] = render_all_rays(rays_o, rays_d, bounds);
  std::cout << "render_all_rays(): " << timer << std::endl;

  Tensor pred_pixels = pred_colors.view({pose_num, batch_size, 3});
  pred_pixels = pred_pixels.clip(0.f, 1.f);
  pred_pixels = pred_pixels.to(image.device());  // (pose_num, batch_size, 3)

  Tensor gt_pixels = image.index({i, j});              // (batch_size, 3)
  Tensor diff = pred_pixels - gt_pixels;               // (pose_num, batch_size, 3)
  Tensor loss = (diff * diff).mean(-1).sum(-1).cpu();  // (pose_num,)

  std::vector<float> result(loss.data_ptr<float>(), loss.data_ptr<float>() + loss.numel());
  return result;
}
