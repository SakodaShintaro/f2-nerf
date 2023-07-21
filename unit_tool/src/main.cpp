#include "../../ros2/src/ros2-f2-nerf/src/localizer_core.hpp"

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
    torch::Tensor optimized_pose = core.optimize_pose(initial_pose, image_tensor, iteration_num);
    std::cout << "Number " << i << std::endl;
    std::cout << "initial_pose\n" << initial_pose << std::endl;
    std::cout << "optimized_pose\n" << optimized_pose << std::endl;
  }
}
