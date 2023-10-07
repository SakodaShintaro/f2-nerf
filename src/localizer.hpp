#ifndef F2_NERF__LOCALIZER_HPP_
#define F2_NERF__LOCALIZER_HPP_

#include "dataset.hpp"
#include "renderer.hpp"

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
  std::vector<Particle> random_search(
    Tensor initial_pose, Tensor image_tensor, int64_t particle_num, float noise_coeff);
  std::vector<Tensor> optimize_pose(
    Tensor initial_pose, Tensor image_tensor, int64_t iteration_num);

  torch::Tensor world2camera(const torch::Tensor & pose_in_world);
  torch::Tensor camera2world(const torch::Tensor & pose_in_camera);

  static Tensor calc_average_pose(const std::vector<Particle> & particles);

  float radius() const { return radius_; }
  int infer_height() const { return infer_height_; }
  int infer_width() const { return infer_width_; }

private:
  std::vector<float> evaluate_poses(const std::vector<Tensor> & poses, const Tensor & image);

  LocalizerCoreParam param_;

  std::shared_ptr<Renderer> renderer_;

  torch::Tensor axis_convert_mat_;

  int infer_height_, infer_width_;
  Tensor intrinsic_;
  Tensor center_;
  float radius_;
};

#endif  // F2_NERF__LOCALIZER_HPP_
