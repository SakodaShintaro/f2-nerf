#include "../../ros2/src/ros2-f2-nerf/src/localizer_core.hpp"
#include "../main_functions.hpp"
#include "../Utils/StopWatch.h"
#include "../Utils/Utils.h"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem::v1;

void test(const std::string & config_path)
{
  torch::NoGradGuard no_grad_guard;
  LocalizerCoreParam param;
  param.runtime_config_path = config_path;
  param.resize_factor = 8;
  LocalizerCore localizer(param);
  const YAML::Node & config = YAML::LoadFile(config_path);

  Dataset dataset(config);
  const std::string save_dir = config["base_exp_dir"].as<std::string>() + "/test_result/";
  fs::create_directories(save_dir);

  Timer timer;
  timer.start();

  for (int32_t i = 0; i < dataset.n_images_; i++) {
    torch::Tensor initial_pose = dataset.poses_[i];
    torch::Tensor image_tensor = dataset.image_tensors_[i];

    image_tensor = localizer.resize_image(image_tensor);

    torch::Tensor nerf_image = localizer.render_image(initial_pose).cpu();
    torch::Tensor diff = nerf_image - image_tensor;
    torch::Tensor loss = (diff * diff).mean(-1).sum();
    torch::Tensor score = (localizer.infer_height() * localizer.infer_width()) / (loss + 1e-6f);

    std::cout << "score[" << i << "] = " << score.item<float>() << std::endl;

    std::stringstream ss;
    ss << save_dir << std::setfill('0') << std::setw(8) << i << ".png";
    Utils::WriteImageTensor(ss.str(), nerf_image);
  }

  std::cout << "\nTime = " << timer.elapsed_seconds() / dataset.n_images_ << std::endl;
}
