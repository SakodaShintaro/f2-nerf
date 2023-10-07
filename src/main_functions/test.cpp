#include "../localizer.hpp"
#include "../stop_watch.hpp"
#include "../utils.hpp"
#include "main_functions.hpp"

#include <experimental/filesystem>
#include <opencv2/core.hpp>

namespace fs = std::experimental::filesystem::v1;

void test(const std::string & train_result_dir, const std::string & dataset_dir)
{
  torch::NoGradGuard no_grad_guard;
  LocalizerCoreParam param;
  param.train_result_dir = train_result_dir;
  param.resize_factor = 8;
  LocalizerCore localizer(param);

  Dataset dataset(dataset_dir);
  const std::string save_dir = train_result_dir + "/test_result/";
  fs::create_directories(save_dir);

  Timer timer;

  float score_sum = 0.0f;
  float time_sum = 0.0f;

  for (int32_t i = 0; i < dataset.n_images; i++) {
    torch::Tensor initial_pose = dataset.poses[i];
    torch::Tensor image_tensor = dataset.images[i];

    image_tensor =
      utils::resize_image(image_tensor, localizer.infer_height(), localizer.infer_width());

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
    utils::write_image_tensor(ss.str(), nerf_image);
  }

  const float average_time = time_sum / dataset.n_images;
  const float average_score = score_sum / dataset.n_images;

  std::ofstream summary(train_result_dir + "/summary.tsv");
  summary << std::fixed;
  summary << "average_time\taverage_score" << std::endl;
  summary << average_time << "\t" << average_score << std::endl;
  std::cout << "\ntime = " << average_time << ", score = " << average_score << std::endl;
}
