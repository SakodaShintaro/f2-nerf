#ifndef LOCALIZER_HPP_
#define LOCALIZER_HPP_

#include "../dataset/dataset.hpp"
#include "../Renderer/Renderer.h"

#include <torch/torch.h>

struct Particle
{
  torch::Tensor pose;  // (3, 4)
  float weight;
};

struct LocalizerCoreParam
{
  std::string runtime_config_path;
  int32_t render_pixel_num = 256;
  float noise_position_x = 0.025f;
  float noise_position_y = 0.025f;
  float noise_position_z = 0.025f;
  float noise_rotation_x = 2.5f;
  float noise_rotation_y = 2.5f;
  float noise_rotation_z = 2.5f;
  int32_t resize_factor = 1;
};

class LocalizerCore
{
  using Tensor = torch::Tensor;

public:
  LocalizerCore() = default;
  LocalizerCore(const LocalizerCoreParam & param);

  Tensor render_image(const Tensor & pose);
  std::tuple<float, Tensor> pred_image_and_calc_score(const Tensor & pose, const Tensor & image);
  std::vector<Particle> random_search(
    Tensor initial_pose, Tensor image_tensor, int64_t particle_num, float noise_coeff);
  std::vector<Tensor> optimize_pose(
    Tensor initial_pose, Tensor image_tensor, int64_t iteration_num);

  torch::Tensor world2camera(const torch::Tensor & pose_in_world);
  torch::Tensor camera2world(const torch::Tensor & pose_in_camera);

  Tensor normalize_position(Tensor pose);
  Tensor inverse_normalize_position(Tensor pose);

  float radius() const { return radius_; }

  Tensor resize_image(Tensor image);

  static Tensor calc_average_pose(const std::vector<Particle> & particles);

  BoundedRays rays_from_pose(const Tensor & pose);

  std::tuple<Tensor, Tensor> render_all_rays(
    const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds);

  int infer_height() const { return infer_height_; }
  int infer_width() const { return infer_width_; }

private:
  std::vector<float> evaluate_poses(const std::vector<Tensor> & poses, const Tensor & image);

  LocalizerCoreParam param_;

  std::shared_ptr<Renderer> renderer_;

  // Convert mat
  torch::Tensor axis_convert_mat1_;

  int n_images_;
  int train_height_, train_width_;
  int infer_height_, infer_width_;
  Tensor intrinsic_;
  float near_, far_;
  Tensor center_;
  float radius_;
};

#endif  // LOCALIZER_HPP_
