#include "localizer.hpp"

#include "dataset.hpp"
#include "stop_watch.hpp"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <gtsam/geometry/Rot3.h>

#include <random>

using Tensor = torch::Tensor;

LocalizerCore::LocalizerCore(const LocalizerCoreParam & param) : param_(param)
{
  cv::FileStorage config(param.runtime_config_path, cv::FileStorage::READ);
  if (!config.isOpened()) {
    throw std::runtime_error("Failed to open " + param.runtime_config_path);
  }

  const std::string base_exp_dir = config["base_exp_dir"].string();
  std::cout << "base_exp_dir: " << base_exp_dir << std::endl;

  cv::FileStorage inference_params(base_exp_dir + "/inference_params.yaml", cv::FileStorage::READ);
  if (!inference_params.isOpened()) {
    throw std::runtime_error("Failed to open " + base_exp_dir + "/inference_params.yaml");
  }

  const int n_images = (int)inference_params["n_images"];
  const int train_height = (int)inference_params["height"];
  const int train_width = (int)inference_params["width"];

  std::vector<float> intrinsic_vector;
  inference_params["intrinsic"] >> intrinsic_vector;
  intrinsic_ = torch::tensor(intrinsic_vector, torch::kFloat).view({3, 3}).to(torch::kCUDA);

  std::vector<float> bounds;
  inference_params["bounds"] >> bounds;
  near_ = bounds[0];
  far_ = bounds[1];

  std::vector<float> normalizing_center;
  inference_params["normalizing_center"] >> normalizing_center;
  center_ = torch::tensor(normalizing_center, torch::kFloat).to(torch::kCUDA);

  radius_ = (float)inference_params["normalizing_radius"];

  const bool use_app_emb = (config["renderer"]["use_app_emb"].string() == "true");
  renderer_ = std::make_shared<Renderer>(use_app_emb, n_images);

  const std::string checkpoint_path = base_exp_dir + "/checkpoints/latest";
  Tensor scalars;
  torch::load(scalars, checkpoint_path + "/scalars.pt");

  torch::load(renderer_, checkpoint_path + "/renderer.pt");

  // set
  infer_height_ = train_height / param.resize_factor;
  infer_width_ = train_width / param.resize_factor;
  std::cout << "infer_height_ = " << infer_height_ << ", infer_width_ = " << infer_width_
            << ", factor = " << param.resize_factor << std::endl;
  intrinsic_ /= param.resize_factor;
  intrinsic_[2][2] = 1.0;
  std::cout << "intrinsic_ = \n" << intrinsic_ << std::endl;

  /*
  [[ 0,  0, -1,  0 ],
  [ -1,  0,  0,  0 ],
  [  0, +1,  0,  0 ],
  [  0,  0,  0, +1 ]]
*/
  axis_convert_mat_ = torch::zeros({4, 4});
  axis_convert_mat_[0][2] = -1;
  axis_convert_mat_[1][0] = -1;
  axis_convert_mat_[2][1] = 1;
  axis_convert_mat_[3][3] = 1;
  axis_convert_mat_ = axis_convert_mat_.to(torch::kCUDA);
}

std::vector<Particle> LocalizerCore::random_search(
  Tensor initial_pose, Tensor image_tensor, int64_t particle_num, float noise_coeff)
{
  torch::NoGradGuard no_grad_guard;

  std::mt19937_64 engine(std::random_device{}());

  // 軸の順番が違うことに注意
  // 世界座標(x: Front, y: Left, z: Up)
  // NeRF座標(x: Right, y: Up, z: Back)
  const float pos_noise_x_in_nerf = param_.noise_position_y * noise_coeff / radius_;
  const float pos_noise_y_in_nerf = param_.noise_position_z * noise_coeff / radius_;
  const float pos_noise_z_in_nerf = param_.noise_position_x * noise_coeff / radius_;
  const float theta_x_in_nerf = param_.noise_rotation_y * noise_coeff;
  const float theta_y_in_nerf = param_.noise_rotation_z * noise_coeff;
  const float theta_z_in_nerf = param_.noise_rotation_x * noise_coeff;

  std::normal_distribution<float> dist_position_x(0.0f, pos_noise_x_in_nerf);
  std::normal_distribution<float> dist_position_y(0.0f, pos_noise_y_in_nerf);
  std::normal_distribution<float> dist_position_z(0.0f, pos_noise_z_in_nerf);
  std::normal_distribution<float> dist_rotation_x(0.0f, theta_x_in_nerf);
  std::normal_distribution<float> dist_rotation_y(0.0f, theta_y_in_nerf);
  std::normal_distribution<float> dist_rotation_z(0.0f, theta_z_in_nerf);

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

std::vector<Tensor> LocalizerCore::optimize_pose(
  Tensor initial_pose, Tensor image_tensor, int64_t iteration_num)
{
  Tensor prev = initial_pose.detach().clone();
  std::vector<Tensor> results;
  initial_pose = initial_pose.requires_grad_(true);
  torch::optim::Adam optimizer({initial_pose}, 1 * 1e-4);
  for (int64_t i = 0; i < iteration_num; i++) {
    auto [rays_o, rays_d, bounds] = rays_from_pose(initial_pose);
    auto [pred_colors, pred_disps] = renderer_->render_all_rays(rays_o, rays_d, bounds);

    Tensor pred_img = pred_colors.view({infer_height_, infer_width_, 3});
    pred_img = pred_img.clip(0.f, 1.f);
    pred_img = pred_img.to(image_tensor.device());

    image_tensor = image_tensor.view({infer_height_, infer_width_, 3});

    Tensor loss = torch::nn::functional::mse_loss(pred_img, image_tensor);
    optimizer.zero_grad();
    // For some reason, backward may fail, so check here
    try {
      loss.backward();
    } catch (const std::runtime_error & e) {
      return results;
    }
    optimizer.step();

    Tensor curr_result = initial_pose.clone().detach();
    curr_result.index_put_({Slc(0, 3), Slc(0, 3)}, prev.index({Slc(0, 3), Slc(0, 3)}));
    results.push_back(curr_result);
  }
  return results;
}

Tensor LocalizerCore::render_image(const Tensor & pose)
{
  torch::NoGradGuard no_grad_guard;
  auto [rays_o, rays_d, bounds] = rays_from_pose(pose);
  auto [image, _] = renderer_->render_all_rays(rays_o, rays_d, bounds);
  image = image.clip(0.0f, 1.0f);
  image = image.view({infer_height_, infer_width_, 3});
  return image;
}

Tensor LocalizerCore::normalize_position(Tensor pose)
{
  Tensor cam_pos = pose.index({Slc(0, 3), 3}).clone();
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  pose.index_put_({Slc(0, 3), 3}, cam_pos);
  return pose;
}

Tensor LocalizerCore::denormalize_position(Tensor pose)
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
  const auto CUDALong = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

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
  std::vector<int> indices(infer_height_ * infer_width_);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 engine(std::random_device{}());
  std::shuffle(indices.begin(), indices.end(), engine);
  std::vector<int64_t> i_vec, j_vec;
  for (int k = 0; k < pixel_num; k++) {
    const int v = indices[k];
    const int64_t i = v / infer_width_;
    const int64_t j = v % infer_width_;
    i_vec.push_back(i);
    j_vec.push_back(j);
  }
  Tensor i = torch::tensor(i_vec, CUDALong);
  Tensor j = torch::tensor(j_vec, CUDALong);

  // Pick rays by random sampling with replacement
  // const Tensor i = torch::randint(0, H, pixel_num, CUDALong);
  // const Tensor j = torch::randint(0, infer_width_, pixel_num, CUDALong);

  const Tensor ij = torch::stack({i, j}, -1).to(torch::kFloat32);
  std::vector<Tensor> rays_o_vec;
  std::vector<Tensor> rays_d_vec;
  for (const Tensor & pose : poses) {
    auto [rays_o, rays_d] = get_rays_from_pose(pose.unsqueeze(0), intrinsic_.unsqueeze(0), ij);
    rays_o_vec.push_back(rays_o);
    rays_d_vec.push_back(rays_d);
  }

  const int64_t pose_num = poses.size();
  const int64_t numel = pixel_num * pose_num;

  Tensor rays_o = torch::cat(rays_o_vec);  // (numel, 3)
  Tensor rays_d = torch::cat(rays_d_vec);  // (numel, 3)
  Tensor bounds =
    torch::stack(
      {torch::full({numel}, near_, CUDAFloat), torch::full({numel}, far_, CUDAFloat)}, -1)
      .contiguous();  // (numel, 2)

  timer.start();
  auto [pred_colors, pred_disps] = renderer_->render_all_rays(rays_o, rays_d, bounds);

  Tensor pred_pixels = pred_colors.view({pose_num, pixel_num, 3});
  pred_pixels = pred_pixels.clip(0.f, 1.f);
  pred_pixels = pred_pixels.to(image.device());  // (pose_num, pixel_num, 3)

  i = i.to(image.device());
  j = j.to(image.device());

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
  Tensor ii = torch::linspace(0.f, infer_height_ - 1.f, infer_height_, CUDAFloat);
  Tensor jj = torch::linspace(0.f, infer_width_ - 1.f, infer_width_, CUDAFloat);
  auto ij = torch::meshgrid({ii, jj}, "ij");
  Tensor i = ij[0].reshape({-1});
  Tensor j = ij[1].reshape({-1});

  auto [rays_o, rays_d] =
    get_rays_from_pose(pose.unsqueeze(0), intrinsic_.unsqueeze(0), torch::stack({i, j}, -1));

  Tensor bounds = torch::stack(
                    {torch::full({infer_height_ * infer_width_}, near_, CUDAFloat),
                     torch::full({infer_height_ * infer_width_}, far_, CUDAFloat)},
                    -1)
                    .contiguous();

  return {rays_o, rays_d, bounds};
}

torch::Tensor LocalizerCore::world2camera(const torch::Tensor & pose_in_world)
{
  torch::Tensor x = pose_in_world;
  x = torch::mm(x, axis_convert_mat_);
  x = torch::mm(axis_convert_mat_.t(), x);
  x = normalize_position(x);
  x = x.index({Slc(0, 3), Slc(0, 4)});
  return x;
}

torch::Tensor LocalizerCore::camera2world(const torch::Tensor & pose_in_camera)
{
  torch::Tensor x = pose_in_camera;
  x = torch::cat({x, torch::tensor({0, 0, 0, 1}).view({1, 4}).to(torch::kCUDA)});
  x = denormalize_position(x);
  x = torch::mm(x, axis_convert_mat_.t());
  x = torch::mm(axis_convert_mat_, x);
  return x;
}
