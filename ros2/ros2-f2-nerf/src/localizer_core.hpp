#ifndef LOCALIZER_CORE_HPP_
#define LOCALIZER_CORE_HPP_

#include "../../src/Dataset/Dataset.h"
#include "../../src/Renderer/Renderer.h"
#include "../../src/Utils/CameraUtils.h"

#include <torch/torch.h>

class LocalizerCore
{
  using Tensor = torch::Tensor;

public:
  LocalizerCore(const std::string & conf_path);

  std::pair<float, Tensor> monte_carlo_localize(Tensor initial_pose, Tensor image_tensor);

private:
  void load_checkpoint(const std::string & checkpoint_path);
  void update_ada_params();
  std::tuple<Tensor, Tensor, Tensor> render_whole_image(
    Tensor rays_o, Tensor rays_d, Tensor bounds);
  float calc_score(Tensor pose, Tensor image);

  BoundedRays rays_from_pose(const Tensor & pose, int reso_level = 1);

  const int H = 227;
  const int W = 388;

  Tensor intri_;
  Tensor dist_params_;

  std::unique_ptr<GlobalDataPool> global_data_pool_;
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<Renderer> renderer_;
};

#endif  // LOCALIZER_CORE_HPP_
