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

  std::tuple<float, Tensor, Tensor> monte_carlo_localize(Tensor initial_pose, Tensor image_tensor);

  Tensor normalize_position(Tensor pose);

  std::unique_ptr<Dataset> dataset_;

private:
  void load_checkpoint(const std::string & checkpoint_path);
  std::tuple<Tensor, Tensor, Tensor> render_whole_image(
    const Tensor & rays_o, const Tensor & rays_d, const Tensor & bounds);
  std::tuple<float, Tensor> calc_score(const Tensor & pose, const Tensor & image);

  BoundedRays rays_from_pose(const Tensor & pose, int reso_level = 1);

  int H;
  int W;

  Tensor intri_;
  Tensor dist_params_;

  std::unique_ptr<GlobalDataPool> global_data_pool_;
  std::unique_ptr<Renderer> renderer_;
};

#endif  // LOCALIZER_CORE_HPP_
