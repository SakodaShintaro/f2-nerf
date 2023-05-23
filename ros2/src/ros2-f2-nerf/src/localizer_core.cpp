#include "localizer_core.hpp"

#include "../../src/Dataset/Dataset.h"
#include "timer.hpp"

#include <Eigen/Geometry>

#include <gtsam/geometry/Rot3.h>

#include <random>

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

std::vector<Particle> LocalizerCore::random_search(
  Tensor initial_pose, Tensor image_tensor, int64_t particle_num)
{
  torch::NoGradGuard no_grad_guard;

  constexpr float NOISE_POSITION = 0.025f;
  constexpr float NOISE_ROTATION = 2.5f;

  std::mt19937_64 engine(std::random_device{}());
  std::normal_distribution<float> dist_position(0.0f, NOISE_POSITION);
  std::normal_distribution<float> dist_rotation(0.0f, NOISE_ROTATION);

  std::vector<Tensor> poses;
  for (int64_t i = 0; i < particle_num; i++) {
    // Sample a random translation
    Tensor curr_pose = initial_pose.clone();
    curr_pose[2][3] += dist_position(engine);
    curr_pose[1][3] += dist_position(engine);
    curr_pose[0][3] += dist_position(engine);

    // orientation
    const float theta_x = dist_rotation(engine) * M_PI / 180.0;
    const float theta_y = dist_rotation(engine) * M_PI / 180.0;
    const float theta_z = dist_rotation(engine) * M_PI / 180.0;
    Eigen::Matrix3f rotation_matrix_x(Eigen::AngleAxisf(theta_x, Eigen::Vector3f::UnitX()));
    Eigen::Matrix3f rotation_matrix_y(Eigen::AngleAxisf(theta_y, Eigen::Vector3f::UnitY()));
    Eigen::Matrix3f rotation_matrix_z(Eigen::AngleAxisf(theta_z, Eigen::Vector3f::UnitZ()));
    const torch::Device dev = initial_pose.device();
    Tensor rotation_tensor_x =
      torch::from_blob(rotation_matrix_x.data(), {3, 3}).to(torch::kFloat32).to(dev);
    Tensor rotation_tensor_y =
      torch::from_blob(rotation_matrix_y.data(), {3, 3}).to(torch::kFloat32).to(dev);
    Tensor rotation_tensor_z =
      torch::from_blob(rotation_matrix_z.data(), {3, 3}).to(torch::kFloat32).to(dev);
    Tensor rotated = rotation_tensor_z.mm(
      rotation_tensor_y.mm(rotation_tensor_x.mm(curr_pose.index({Slc(0, 3), Slc(0, 3)}))));
    curr_pose.index_put_({Slc(0, 3), Slc(0, 3)}, rotated);
    poses.push_back(curr_pose);
  }

  const std::vector<float> weights = evaluate_poses(poses, image_tensor);
  const int pose_num = poses.size();

  std::vector<Particle> result;
  for (int i = 0; i < pose_num; i++) {
    result.push_back({poses[i], weights[i]});
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

  const int ray_batch_size = (1 << 16);
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

std::tuple<float, Tensor> LocalizerCore::pred_image_and_calc_score(
  const Tensor & pose, const Tensor & image)
{
  torch::NoGradGuard no_grad_guard;
  Timer timer;
  auto [rays_o, rays_d, bounds] = dataset_->RaysFromPose(pose);
  timer.reset();
  auto [pred_colors, first_oct_dis, pred_disps] = render_all_rays(rays_o, rays_d, bounds);

  Tensor pred_img = pred_colors.view({H, W, 3});
  pred_img = pred_img.clip(0.f, 1.f);
  pred_img = pred_img.to(image.device());

  Tensor diff = pred_img - image.view({H, W, 3});
  Tensor loss = (diff * diff).mean(-1).sum();
  Tensor score = (H * W) / (loss + 1e-6f);
  return {score.mean().item<float>(), pred_img};
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
  const int batch_size = 256;

  // Pick rays by constant interval
  // const int step = H * W / batch_size;
  // std::vector<int64_t> i_vec, j_vec;
  // for (int k = 0; k < batch_size; k++) {
  //   const int v = k * step;
  //   const int64_t i = v / W;
  //   const int64_t j = v % W;
  //   i_vec.push_back(i);
  //   j_vec.push_back(j);
  // }
  // const Tensor i = torch::tensor(i_vec, CUDALong);
  // const Tensor j = torch::tensor(j_vec, CUDALong);

  // Pick rays by random sampling without replacement
  std::vector<int> indices(H * W);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 engine(std::random_device{}());
  std::shuffle(indices.begin(), indices.end(), engine);
  std::vector<int64_t> i_vec, j_vec;
  for (int k = 0; k < batch_size; k++) {
    const int v = indices[k];
    const int64_t i = v / W;
    const int64_t j = v % W;
    i_vec.push_back(i);
    j_vec.push_back(j);
  }
  const Tensor i = torch::tensor(i_vec, CUDALong);
  const Tensor j = torch::tensor(j_vec, CUDALong);

  // Pick rays by random sampling with replacement
  // const Tensor i = torch::randint(0, H, batch_size, CUDALong);
  // const Tensor j = torch::randint(0, W, batch_size, CUDALong);

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

  timer.reset();
  auto [pred_colors, first_oct_dis, pred_disps] = render_all_rays(rays_o, rays_d, bounds);

  Tensor pred_pixels = pred_colors.view({pose_num, batch_size, 3});
  pred_pixels = pred_pixels.clip(0.f, 1.f);
  pred_pixels = pred_pixels.to(image.device());  // (pose_num, batch_size, 3)

  Tensor gt_pixels = image.index({i, j});              // (batch_size, 3)
  Tensor diff = pred_pixels - gt_pixels;               // (pose_num, batch_size, 3)
  Tensor loss = (diff * diff).mean(-1).sum(-1).cpu();  // (pose_num,)
  loss = batch_size / (loss + 1e-6f);
  loss = torch::pow(loss, 5);
  loss /= loss.sum();

  std::vector<float> result(loss.data_ptr<float>(), loss.data_ptr<float>() + loss.numel());
  return result;
}

Eigen::Matrix3d compute_rotation_average(
  const std::vector<Eigen::Matrix3d> & rotations, const std::vector<double> & weights)
{
  const double epsilon = 0.000001;
  const int max_iters = 300;
  Eigen::Matrix3d R = rotations[0];
  for (int iter = 0; iter < max_iters; ++iter) {
    Eigen::Vector3d rot_sum = Eigen::Vector3d::Zero();
    for (int i = 0; i < rotations.size(); ++i) {
      const Eigen::Matrix3d & rot = rotations[i];
      gtsam::Rot3 g_rot = gtsam::Rot3(R.transpose() * rot);
      rot_sum += weights[i] * gtsam::Rot3::Logmap(g_rot);
    }

    if (rot_sum.norm() < epsilon) {
      return R;
    } else {
      Eigen::Matrix3d r = gtsam::Rot3::Expmap(rot_sum).matrix();
      Eigen::Matrix3d s = R * r;
      R = gtsam::Rot3(s).matrix();
    }
  }
  return R;
}

Tensor LocalizerCore::calc_average_pose(const std::vector<Particle> & particles)
{
  torch::Device device = particles.front().pose.device();
  torch::Tensor avg_position_tensor = torch::zeros({3, 1}, device).to(torch::kFloat32);
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<double> weights;

  for (const Particle & particle : particles) {
    torch::Tensor pose = particle.pose;
    torch::Tensor position = pose.index({Slc(0, 3), Slc(3, 4)});
    avg_position_tensor += position * particle.weight;

    // slice to get 3x3 rotation matrix, convert it to Eigen::Matrix3f
    torch::Tensor rotation_tensor = pose.index({Slc(0, 3), Slc(0, 3)}).to(torch::kDouble).cpu();
    Eigen::Matrix3d rotation;
    std::memcpy(
      rotation.data(), rotation_tensor.data_ptr(), sizeof(double) * rotation_tensor.numel());
    rotations.push_back(rotation);
    weights.push_back(particle.weight);
  }

  Eigen::Matrix3d avg_rotation_matrix = compute_rotation_average(rotations, weights);
  torch::Tensor avg_rotation_tensor = torch::from_blob(
    avg_rotation_matrix.data(), {3, 3}, torch::TensorOptions().dtype(torch::kDouble));
  avg_rotation_tensor = avg_rotation_tensor.to(torch::kFloat32);
  avg_rotation_tensor = avg_rotation_tensor.to(device);

  // combine average position and rotation to form average pose
  torch::Tensor avg_pose = torch::zeros_like(particles.front().pose);
  avg_pose.index_put_({Slc(0, 3), Slc(3, 4)}, avg_position_tensor);
  avg_pose.index_put_({Slc(0, 3), Slc(0, 3)}, avg_rotation_tensor);

  return avg_pose;
}
