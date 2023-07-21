#include "../../ros2/src/ros2-f2-nerf/src/localizer_core.hpp"
#include "../../src/Utils/Utils.h"

int main()
{
  std::cout << "unit tool" << std::endl;
  const std::string runtime_config_path = "../../ros2/runtime_config_awsim.yaml";

  LocalizerCoreParam param{};
  param.is_awsim = true;
  LocalizerCore core(runtime_config_path, param);

  constexpr int32_t iteration_num = 1;

  for (int32_t i = 0; i < core.dataset_->n_images_; i++) {
    torch::Tensor initial_pose = core.dataset_->poses_[i];
    torch::Tensor image_tensor = core.dataset_->image_tensors_[i];
    Utils::WriteImageTensor("image_01_gt.png", image_tensor);

    auto [score_before, nerf_image_before] =
      core.pred_image_and_calc_score(initial_pose, image_tensor);

    std::cout << "Number " << i << std::endl;
    std::cout << "score_before\n" << score_before << std::endl;
    Utils::WriteImageTensor("image_02_before.png", nerf_image_before);

    torch::Tensor optimized_pose = core.optimize_pose(initial_pose, image_tensor, iteration_num);
    std::cout << "optimized_pose\n" << optimized_pose << std::endl;
    auto [score_after, nerf_image_after] =
      core.pred_image_and_calc_score(optimized_pose, image_tensor);
    std::cout << "score_after\n" << score_after << std::endl;
    Utils::WriteImageTensor("image_03_after.png", nerf_image_after);

    break;
  }
}
