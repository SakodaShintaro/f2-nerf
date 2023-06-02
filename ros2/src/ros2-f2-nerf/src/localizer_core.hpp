#ifndef LOCALIZER_CORE_HPP_
#define LOCALIZER_CORE_HPP_

#include "../../src/Dataset/Dataset.h"
#include "../../src/Renderer/Renderer.h"
#include "../../src/Utils/CameraUtils.h"

#include <torch/torch.h>

struct Particle
{
  torch::Tensor pose;  // (3, 4)
  float weight;
};

struct LocalizerCoreParam
{
  int32_t render_pixel_num = 256;
  float noise_position_x = 0.025f;
  float noise_position_y = 0.025f;
  float noise_position_z = 0.025f;
  float noise_rotation_x = 2.5f;
  float noise_rotation_y = 2.5f;
  float noise_rotation_z = 2.5f;
  bool is_awsim;
};

class LocalizerCore
{
  using Tensor = torch::Tensor;

public:
  LocalizerCore() = default;
  LocalizerCore(const std::string & conf_path, const LocalizerCoreParam & param);

  std::tuple<float, Tensor> pred_image_and_calc_score(const Tensor & pose, const Tensor & image);
  std::vector<Particle> random_search(
    Tensor initial_pose, Tensor image_tensor, int64_t particle_num, float noise_coeff);

  Tensor normalize_position(Tensor pose);
  Tensor inverse_normalize_position(Tensor pose);

  static Tensor calc_average_pose(const std::vector<Particle> & particles);

  std::unique_ptr<Dataset> dataset_;

  int H;
  int W;

private:
  void load_checkpoint(const std::string & checkpoint_path);
  std::tuple<Tensor, Tensor, Tensor> render_all_rays(
    const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds);
  std::vector<float> evaluate_poses(const std::vector<Tensor> & poses, const Tensor & image);

  LocalizerCoreParam param_;

  std::unique_ptr<GlobalDataPool> global_data_pool_;
  std::unique_ptr<Renderer> renderer_;
};

#endif  // LOCALIZER_CORE_HPP_
