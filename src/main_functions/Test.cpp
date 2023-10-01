#include "../utils/stop_watch.hpp"
#include "../utils/utils.hpp"
#include "../localizer/localizer.hpp"
#include "../main_functions.hpp"

#include <experimental/filesystem>
#include <opencv2/core.hpp>

namespace fs = std::experimental::filesystem::v1;

void test(const std::string & config_path)
{
  torch::NoGradGuard no_grad_guard;
  LocalizerCoreParam param;
  param.runtime_config_path = config_path;
  param.resize_factor = 8;
  LocalizerCore localizer(param);
  cv::FileStorage config(config_path, cv::FileStorage::READ);

  const std::string data_path = config["dataset_path"].string();
  const std::string base_exp_dir = config["base_exp_dir"].string();

  Dataset dataset(data_path, base_exp_dir);
  const std::string save_dir = config["base_exp_dir"].string() + "/test_result/";
  fs::create_directories(save_dir);

  Timer timer;

  float score_sum = 0.0f;
  float time_sum = 0.0f;

  for (int32_t i = 0; i < dataset.n_images_; i++) {
    torch::Tensor initial_pose = dataset.poses_[i];
    torch::Tensor image_tensor = dataset.image_tensors_[i];

    image_tensor = localizer.resize_image(image_tensor);

    timer.start();
    torch::Tensor nerf_image = localizer.render_image(initial_pose).cpu();
    time_sum += timer.elapsed_seconds();
    torch::Tensor diff = nerf_image - image_tensor;
    torch::Tensor loss = (diff * diff).mean(-1).sum();
    torch::Tensor score = (localizer.infer_height() * localizer.infer_width()) / (loss + 1e-6f);

    std::cout << "\rscore[" << i << "] = " << score.item<float>() << std::flush;
    score_sum += score.item<float>();

    std::stringstream ss;
    ss << save_dir << std::setfill('0') << std::setw(8) << i << ".png";
    Utils::WriteImageTensor(ss.str(), nerf_image);
  }

  const float average_time = time_sum / dataset.n_images_;
  const float average_score = score_sum / dataset.n_images_;

  std::ofstream summary(base_exp_dir + "/summary.tsv");
  summary << std::fixed;
  summary << "average_time\taverage_score" << std::endl;
  summary << average_time << "\t" << average_score << std::endl;
  std::cout << "\ntime = " << average_time << ", score = " << average_score << std::endl;
}
