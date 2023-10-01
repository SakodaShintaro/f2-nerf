#include "../localizer.hpp"
#include "../stop_watch.hpp"
#include "../utils.hpp"
#include "main_functions.hpp"

#include <experimental/filesystem>
#include <opencv2/core.hpp>

namespace fs = std::experimental::filesystem::v1;

enum Dir { kUp, kUpRight, kRight, kDownRight, kDown, kDownLeft, kLeft, kUpLeft, kDirNum };
constexpr int64_t kDx[kDirNum] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int64_t kDz[kDirNum] = {1, 1, 0, -1, -1, -1, 0, 1};

void infer(const std::string & config_path)
{
  LocalizerCoreParam param{};
  param.resize_factor = 32;
  param.runtime_config_path = config_path;
  LocalizerCore core(param);

  constexpr int32_t iteration_num = 10;

  cv::FileStorage config(config_path, cv::FileStorage::READ);
  const std::string data_path = config["dataset_path"].string();
  const std::string base_exp_dir = config["base_exp_dir"].string();

  const std::string save_dir = base_exp_dir + "/inference_result/";
  fs::create_directories(save_dir);

  Dataset dataset(data_path, base_exp_dir);

  Timer timer, timer_local;
  timer.start();

  std::vector<double> optimize_times;

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
      timer_local.start();
      std::vector<torch::Tensor> optimized_poses =
        core.optimize_pose(curr_pose, image_tensor, iteration_num);
      optimize_times.push_back(timer_local.elapsed_seconds());
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
  }

  torch::Tensor optimize_time_tensor = torch::tensor(optimize_times, torch::kFloat);

  std::cout << "\nAverage Time = " << optimize_time_tensor.mean().item<float>() << " sec"
            << std::endl;
}
