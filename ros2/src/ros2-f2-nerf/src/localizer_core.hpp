#ifndef LOCALIZER_CORE_HPP_
#define LOCALIZER_CORE_HPP_

#include "../../src/Dataset/Dataset.h"
#include "../../src/Renderer/Renderer.h"
#include "../../src/Utils/CameraUtils.h"

#include <torch/torch.h>

struct Particle
{
  torch::Tensor pose;  // (4, 4)
  float weight;
};

class LocalizerCore
{
  using Tensor = torch::Tensor;

public:
  LocalizerCore(const std::string & conf_path);

  std::tuple<float, Tensor> pred_image_and_calc_score(const Tensor & pose, const Tensor & image);
  std::vector<Particle> grid_search(Tensor initial_pose, Tensor image_tensor);

  Tensor normalize_position(Tensor pose);
  Tensor inverse_normalize_position(Tensor pose);

  static Tensor calc_average_pose(const std::vector<Particle> & particles);

  std::unique_ptr<Dataset> dataset_;

private:
  void load_checkpoint(const std::string & checkpoint_path);
  std::tuple<Tensor, Tensor, Tensor> render_all_rays(
    const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds);
  std::vector<float> evaluate_poses(const std::vector<Tensor> & poses, const Tensor & image);

  int H;
  int W;

  std::unique_ptr<GlobalDataPool> global_data_pool_;
  std::unique_ptr<Renderer> renderer_;
};

#endif  // LOCALIZER_CORE_HPP_
