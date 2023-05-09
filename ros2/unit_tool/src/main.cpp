#include "../../ros2-f2-nerf/src/localizer_core.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

int main(int argc, char * argv[])
{
  std::cout << "unit_tool" << std::endl;
  LocalizerCore localizer_core("./runtime_config.yaml");

  cv::Mat image =
    cv::imread("/root/f2-nerf/data/converted/20230503_raw_colmap_try1/images/00000000.png");

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
  image_tensor = image_tensor.view({1, height, width, channels});
  image_tensor = image_tensor.permute({0, 3, 1, 2});

  Eigen::Quaternionf quat(1, 0, 0, 0);
  Eigen::Matrix3f rot = quat.toRotationMatrix();

  torch::Tensor initial_pose = torch::zeros({3, 4});
  initial_pose[0][0] = 0.9856;
  initial_pose[0][1] = 0.0329;
  initial_pose[0][2] = -0.1660;
  initial_pose[0][3] = 0.4609;
  initial_pose[1][0] = 0.0328;
  initial_pose[1][1] = -0.9995;
  initial_pose[1][2] = 0.0036;
  initial_pose[1][3] = 0.0287;
  initial_pose[2][0] = -0.1660;
  initial_pose[2][1] = -0.0019;
  initial_pose[2][2] = -0.9861;
  initial_pose[2][3] = -0.3729;

  std::cout << initial_pose << std::endl;

  image_tensor = image_tensor.to(CUDAFloat).contiguous();
  initial_pose = initial_pose.to(CUDAFloat).contiguous();

  // run NeRF
  const auto [score, optimized_pose] =
    localizer_core.monte_carlo_localize(initial_pose, image_tensor);
}
