#include "../ros2/src/ros2-f2-nerf/src/localizer_core.hpp"
#include "../src/Utils/StopWatch.h"
#include "../src/Utils/Utils.h"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem::v1;

enum Dir { kUp, kUpRight, kRight, kDownRight, kDown, kDownLeft, kLeft, kUpLeft, kDirNum };
constexpr int64_t kDx[kDirNum] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int64_t kDz[kDirNum] = {1, 1, 0, -1, -1, -1, 0, 1};

void infer(const std::string & config_path)
{
  LocalizerCoreParam param{};
  param.resize_factor = 16;
  param.runtime_config_path = config_path;
  LocalizerCore core(param);

  constexpr int32_t iteration_num = 10;

  const std::string save_dir = "./inference_result/";
  fs::create_directories(save_dir);

  const YAML::Node & config = YAML::LoadFile(config_path);
  Dataset dataset(config);

  Timer timer;
  timer.start();

  const float noise = 0.5f / core.radius();
  std::cout << "noise = " << noise << std::endl;

  for (int32_t i = 0; i < dataset.n_images_; i++) {
    std::cout << "\rTime " << static_cast<int64_t>(timer.elapsed_seconds()) << " " << i << "/"
              << dataset.n_images_ << std::flush;
    const std::string curr_dir =
      (std::stringstream() << save_dir << "/" << std::setfill('0') << std::setw(4) << i << "/")
        .str();
    fs::create_directories(curr_dir);

    torch::Tensor initial_pose = dataset.poses_[i];
    torch::Tensor image_tensor = dataset.image_tensors_[i];

    image_tensor = core.resize_image(image_tensor);
    Utils::WriteImageTensor(curr_dir + "image_01_gt.png", image_tensor);

    std::ofstream ofs(curr_dir + "/position.tsv");
    ofs << std::fixed;
    ofs << "name\tx\ty\tz\tscore" << std::endl;
    auto output = [&](const std::string & name, const torch::Tensor & pose, float score) {
      const torch::Tensor out = core.camera2world(pose);
      ofs << name << "\t";
      ofs << out[0][3].item<float>() << "\t";
      ofs << out[1][3].item<float>() << "\t";
      ofs << out[2][3].item<float>() << "\t";
      ofs << score << std::endl;
    };

    // Before noise
    auto [score_before, nerf_image_before] =
      core.pred_image_and_calc_score(initial_pose, image_tensor);
    Utils::WriteImageTensor(curr_dir + "image_02_before.png", nerf_image_before);
    output("original", initial_pose, score_before);

    // Added noise
    for (int32_t d = 0; d < kDirNum; d++) {
      torch::Tensor curr_pose = initial_pose.clone();
      curr_pose[0][3] += noise * kDx[d];
      curr_pose[2][3] += noise * kDz[d];
      auto [score_noised, nerf_image_noised] =
        core.pred_image_and_calc_score(curr_pose, image_tensor);
      Utils::WriteImageTensor(
        curr_dir + "image_03_noised" + std::to_string(d) + ".png", nerf_image_noised);
      output("noised_" + std::to_string(d), curr_pose, score_noised);

      // Optimize
      std::vector<torch::Tensor> optimized_poses =
        core.optimize_pose(curr_pose, image_tensor, iteration_num);
      for (int32_t itr = 0; itr < optimized_poses.size(); itr++) {
        torch::Tensor optimized_pose = optimized_poses[itr];
        auto [score_after, nerf_image_after] =
          core.pred_image_and_calc_score(optimized_pose, image_tensor);
        const std::string suffix =
          (std::stringstream() << d << "_" << std::setfill('0') << std::setw(2) << itr).str();
        Utils::WriteImageTensor(curr_dir + "image_04_after_" + suffix + ".png", nerf_image_after);
        output("optimized_" + suffix, optimized_pose, score_after);
      }
    }

    break;
  }

  std::cout << "\nTime = " << timer.elapsed_seconds() << std::endl;
}
