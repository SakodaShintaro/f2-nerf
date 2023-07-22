#include "localizer_core.hpp"

#include "../../src/Dataset/Dataset.h"
#include "timer.hpp"

#include <Eigen/Geometry>

#include <gtsam/geometry/Rot3.h>

#include <random>

using Tensor = torch::Tensor;

LocalizerCore::LocalizerCore(const std::string & conf_path, const LocalizerCoreParam & param)
: param_(param)
{
  const YAML::Node & config = YAML::LoadFile(conf_path);
  const std::string base_exp_dir = config["base_exp_dir"].as<std::string>();
  std::cout << "base_exp_dir: " << base_exp_dir << std::endl;

  const YAML::Node inference_params = YAML::LoadFile(base_exp_dir + "/inference_params.yaml");
  n_images_ = inference_params["n_images"].as<int>();
  height_ = inference_params["height"].as<int>();
  width_ = inference_params["width"].as<int>();
  intrinsic_ = torch::tensor(inference_params["intrinsic"].as<std::vector<float>>(), CUDAFloat)
                 .view({3, 3});
  std::vector<float> bounds = inference_params["bounds"].as<std::vector<float>>();
  near_ = bounds[0];
  far_ = bounds[1];
  center_ =
    torch::tensor(inference_params["normalizing_center"].as<std::vector<float>>(), CUDAFloat);
  radius_ = inference_params["normalizing_radius"].as<float>();

  renderer_ = std::make_unique<Renderer>(config, n_images_);

  const std::string checkpoint_path = base_exp_dir + "/checkpoints/latest";
  Tensor scalars;
  torch::load(scalars, checkpoint_path + "/scalars.pt");

  std::vector<Tensor> scene_states;
  torch::load(scene_states, checkpoint_path + "/renderer.pt");
  renderer_->LoadStates(scene_states, 0);

  // set
  H = height_ / param.resize_factor;
  W = width_ / param.resize_factor;
  std::cout << "H = " << H << ", W = " << W << ", factor = " << param.resize_factor << std::endl;
  intrinsic_ /= param.resize_factor;
  intrinsic_[2][2] = 1.0;
}

std::vector<Particle> LocalizerCore::random_search(
  Tensor initial_pose, Tensor image_tensor, int64_t particle_num, float noise_coeff)
{
  torch::NoGradGuard no_grad_guard;

  std::mt19937_64 engine(std::random_device{}());
  std::normal_distribution<float> dist_position_x(0.0f, param_.noise_position_x * noise_coeff);
  std::normal_distribution<float> dist_position_y(0.0f, param_.noise_position_y * noise_coeff);
  std::normal_distribution<float> dist_position_z(0.0f, param_.noise_position_z * noise_coeff);
  std::normal_distribution<float> dist_rotation_x(0.0f, param_.noise_rotation_x * noise_coeff);
  std::normal_distribution<float> dist_rotation_y(0.0f, param_.noise_rotation_y * noise_coeff);
  std::normal_distribution<float> dist_rotation_z(0.0f, param_.noise_rotation_z * noise_coeff);

  std::vector<Tensor> poses;
  for (int64_t i = 0; i < particle_num; i++) {
    // Sample a random translation
    Tensor curr_pose = initial_pose.clone();
    if (i == 0) {
      poses.push_back(curr_pose);
      continue;
    }
    curr_pose[0][3] += dist_position_x(engine);
    curr_pose[1][3] += dist_position_y(engine);
    curr_pose[2][3] += dist_position_z(engine);

    // orientation
    const float theta_x = dist_rotation_x(engine) * M_PI / 180.0;
    const float theta_y = dist_rotation_y(engine) * M_PI / 180.0;
    const float theta_z = dist_rotation_z(engine) * M_PI / 180.0;
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

torch::Tensor gram_schmidt(torch::Tensor A)
{
  A = A.clone();
  for (int i = 0; i < A.size(0); ++i) {
    for (int j = 0; j < i; ++j) {
      A[i] -= torch::dot(A[j], A[i]) * A[j];
    }
    A[i] = A[i] / A[i].norm();
  }
  return A;
}

Tensor LocalizerCore::optimize_pose(Tensor initial_pose, Tensor image_tensor, int64_t iteration_num)
{
  for (int64_t i = 0; i < iteration_num; i++) {
    initial_pose = initial_pose.requires_grad_(true);
    torch::optim::SGD optimizer({initial_pose}, 1e2);
    auto [rays_o, rays_d, bounds] = rays_from_pose(initial_pose);
    auto [pred_colors, first_oct_dis, pred_disps] = render_all_rays_grad(rays_o, rays_d, bounds);

    Tensor pred_img = pred_colors.view({H, W, 3});
    pred_img = pred_img.clip(0.f, 1.f);
    pred_img = pred_img.to(image_tensor.device());

    Tensor diff = pred_img - image_tensor.view({H, W, 3});
    Tensor loss = (diff * diff).mean(-1).sum();
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // orthogonalize
    initial_pose = initial_pose.detach();
    Tensor rotation = initial_pose.index({Slc(0, 3), Slc(0, 3)});
    rotation = gram_schmidt(rotation);
    initial_pose.index_put_({Slc(0, 3), Slc(0, 3)}, rotation);
  }
  return initial_pose;
}

std::tuple<Tensor, Tensor, Tensor> LocalizerCore::render_all_rays(
  const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds)
{
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

    auto render_result =
      renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor(), RunningMode::VALIDATE);
    Tensor colors = render_result.colors.to(torch::kCPU);
    Tensor disp = render_result.disparity.to(torch::kCPU).squeeze();

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
    if (!render_result.first_oct_dis.sizes().empty()) {
      Tensor & ret_first_oct_dis = render_result.first_oct_dis;
      if (ret_first_oct_dis.has_storage()) {
        Tensor cur_first_oct_dis = render_result.first_oct_dis.to(torch::kCPU);
        first_oct_disp.index_put_({Slc(i, i_high)}, cur_first_oct_dis);
      }
    }
  }
  pred_disp = pred_disp / pred_disp.max();
  first_oct_disp = first_oct_disp.min() / first_oct_disp;

  return {pred_colors, first_oct_disp, pred_disp};
}

std::tuple<Tensor, Tensor, Tensor> LocalizerCore::render_all_rays_grad(
  const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds)
{
  const int n_rays = rays_d.sizes()[0];

  std::vector<Tensor> pred_colors;
  std::vector<Tensor> first_oct_disp;
  std::vector<Tensor> pred_disp;

  const int ray_batch_size = (1 << 16);
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o = rays_o.index({Slc(i, i_high)}).contiguous();
    Tensor cur_rays_d = rays_d.index({Slc(i, i_high)}).contiguous();
    Tensor cur_bounds = bounds.index({Slc(i, i_high)}).contiguous();

    auto render_result =
      renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor(), RunningMode::VALIDATE);
    Tensor colors = render_result.colors;
    Tensor disp = render_result.disparity.squeeze();

    pred_colors.push_back(colors);
    pred_disp.push_back(disp.unsqueeze(-1));
    if (!render_result.first_oct_dis.sizes().empty()) {
      Tensor & ret_first_oct_dis = render_result.first_oct_dis;
      if (ret_first_oct_dis.has_storage()) {
        Tensor cur_first_oct_dis = render_result.first_oct_dis;
        first_oct_disp.push_back(cur_first_oct_dis);
      }
    }
  }

  Tensor pred_colors_ts = torch::cat(pred_colors, 0);
  Tensor pred_disp_ts = torch::cat(pred_disp, 0);
  Tensor first_oct_disp_ts = torch::cat(first_oct_disp, 0);

  pred_disp_ts = pred_disp_ts / pred_disp_ts.max();
  first_oct_disp_ts = first_oct_disp_ts.min() / first_oct_disp_ts;

  return {pred_colors_ts, first_oct_disp_ts, pred_disp_ts};
}

std::tuple<float, Tensor> LocalizerCore::pred_image_and_calc_score(
  const Tensor & pose, const Tensor & image)
{
  torch::NoGradGuard no_grad_guard;
  Timer timer;
  auto [rays_o, rays_d, bounds] = rays_from_pose(pose);
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
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  pose.index_put_({Slc(0, 3), 3}, cam_pos);
  return pose;
}

Tensor LocalizerCore::inverse_normalize_position(Tensor pose)
{
  Tensor cam_pos = pose.index({Slc(0, 3), 3}).clone();
  cam_pos = cam_pos * radius_ + center_.unsqueeze(0);
  pose.index_put_({Slc(0, 3), 3}, cam_pos);
  return pose;
}

std::vector<float> LocalizerCore::evaluate_poses(
  const std::vector<Tensor> & poses, const Tensor & image)
{
  torch::NoGradGuard no_grad_guard;
  Timer timer;

  const int pixel_num = param_.render_pixel_num;

  // Pick rays by constant interval
  // const int step = H * W / pixel_num;
  // std::vector<int64_t> i_vec, j_vec;
  // for (int k = 0; k < pixel_num; k++) {
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
  for (int k = 0; k < pixel_num; k++) {
    const int v = indices[k];
    const int64_t i = v / W;
    const int64_t j = v % W;
    i_vec.push_back(i);
    j_vec.push_back(j);
  }
  const Tensor i = torch::tensor(i_vec, CUDALong);
  const Tensor j = torch::tensor(j_vec, CUDALong);

  // Pick rays by random sampling with replacement
  // const Tensor i = torch::randint(0, H, pixel_num, CUDALong);
  // const Tensor j = torch::randint(0, W, pixel_num, CUDALong);

  const Tensor ij = torch::stack({i, j}, -1).to(torch::kFloat32);
  std::vector<Tensor> rays_o_vec;
  std::vector<Tensor> rays_d_vec;
  for (const Tensor & pose : poses) {
    auto [rays_o, rays_d] = Dataset::Img2WorldRay(pose, intrinsic_, ij);
    rays_o_vec.push_back(rays_o);
    rays_d_vec.push_back(rays_d);
  }

  const int64_t pose_num = poses.size();
  const int64_t numel = pixel_num * pose_num;

  Tensor rays_o = torch::cat(rays_o_vec);  // (numel, 3)
  Tensor rays_d = torch::cat(rays_d_vec);  // (numel, 3)
  Tensor bounds =
    torch::stack({torch::full({numel}, near_, CUDAFloat), torch::full({numel}, far_, CUDAFloat)}, -1)
      .contiguous();  // (numel, 2)

  timer.reset();
  auto [pred_colors, first_oct_dis, pred_disps] = render_all_rays(rays_o, rays_d, bounds);

  Tensor pred_pixels = pred_colors.view({pose_num, pixel_num, 3});
  pred_pixels = pred_pixels.clip(0.f, 1.f);
  pred_pixels = pred_pixels.to(image.device());  // (pose_num, pixel_num, 3)

  Tensor gt_pixels = image.index({i, j});              // (pixel_num, 3)
  Tensor diff = pred_pixels - gt_pixels;               // (pose_num, pixel_num, 3)
  Tensor loss = (diff * diff).mean(-1).sum(-1).cpu();  // (pose_num,)
  loss = pixel_num / (loss + 1e-6f);
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

BoundedRays LocalizerCore::rays_from_pose(const Tensor & pose)
{
  Tensor ii = torch::linspace(0.f, height_ - 1.f, H, CUDAFloat);
  Tensor jj = torch::linspace(0.f, width_ - 1.f, W, CUDAFloat);
  auto ij = torch::meshgrid({ii, jj}, "ij");
  Tensor i = ij[0].reshape({-1});
  Tensor j = ij[1].reshape({-1});

  auto [rays_o, rays_d] = Dataset::Img2WorldRay(pose, intrinsic_, torch::stack({i, j}, -1));

  Tensor bounds =
    torch::stack({torch::full({H * W}, near_, CUDAFloat), torch::full({H * W}, far_, CUDAFloat)}, -1)
      .contiguous();

  return {rays_o, rays_d, bounds};
}
