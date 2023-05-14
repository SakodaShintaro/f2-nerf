#include "../../ros2-f2-nerf/src/localizer_core.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

class Timer
{
public:
  Timer() { start(); }
  void start() { start_time_ = std::chrono::steady_clock::now(); }
  double elapsedMicroSeconds() const
  {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    double microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return microseconds;
  }
  int64_t elapsedSeconds() const
  {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    int64_t seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    return seconds;
  }

private:
  std::chrono::steady_clock::time_point start_time_;
};

int main(int argc, char * argv[])
{
  std::cout << "unit_tool" << std::endl;
  LocalizerCore localizer_core("./runtime_config.yaml");

  cv::Mat image =
    cv::imread("/root/f2-nerf/data/converted/20230501_try1/images/00000000.png");

  // Accessing image data
  const int height = image.rows;
  const int width = image.cols;
  const int channels = image.channels();
  std::cout << "height: " << height << std::endl;
  std::cout << "width: " << width << std::endl;
  std::cout << "channels: " << channels << std::endl;
  std::vector<uint8_t> data(height * width * channels, 0);
  std::copy(image.data, image.data + image.total() * image.elemSize(), data.data());
  torch::Tensor image_tensor = torch::tensor(data);
  image_tensor = image_tensor.view({height, width, channels});
  image_tensor = image_tensor.to(torch::kFloat32);
  image_tensor /= 255.0;

  torch::Tensor initial_pose = localizer_core.dataset_->poses_[0];

  image_tensor = image_tensor.to(CUDAFloat).contiguous();
  initial_pose = initial_pose.to(CUDAFloat).contiguous();

  std::cout << initial_pose << std::endl;

  // run NeRF
  Timer timer;
  constexpr int TRIAL_NUM = 100;
  for (int i = 0; i < TRIAL_NUM; i++) {
    const auto [score, optimized_pose, pred_image] =
      localizer_core.monte_carlo_localize(initial_pose, image_tensor);
  }
  const double average_microsec = timer.elapsedMicroSeconds() / TRIAL_NUM;
  const double average_millisec = average_microsec / 1000;
  std::cout << std::fixed << std::setprecision(1) << std::endl;
  std::cout << "Average = " << average_millisec << " msec" << std::endl;
}
