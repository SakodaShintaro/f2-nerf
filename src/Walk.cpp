#include "Walk.hpp"

#include "../ros2/src/ros2-f2-nerf/src/localizer_core.hpp"
#include "../src/Utils/Utils.h"

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

int kbhit(void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF) {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

torch::Tensor calc_rotation_tensor(float degree, Eigen::Vector3f axis)
{
  const float theta = degree * M_PI / 180.0;
  Eigen::AngleAxisf a(theta, axis);
  Eigen::Matrix3f rotation_matrix(a);
  torch::Tensor result =
    torch::from_blob(rotation_matrix.data(), {3, 3}).to(torch::kFloat32).cuda();
  return result;
}

void walk(const std::string & config_path)
{
  torch::NoGradGuard no_grad_guard;
  LocalizerCoreParam param;
  param.runtime_config_path = config_path;
  param.resize_factor = 16;
  LocalizerCore localizer(param);
  torch::Tensor pose = torch::eye(4).cuda();
  constexpr float step = 0.1;
  constexpr float degree = 10.0;

  // Poseは世界座標系で考える
  // つまりXが前方
  // Yが左方
  // Zが上方
  std::cout << "pose:\n" << pose << std::endl;
  std::cout << "WASDで移動, E:上昇, Q下降, J:左回転, K:下回転, L:右回転, I:上回転" << std::endl;

  while (1) {
    if (kbhit()) {
      const char pushed_key = getchar();
      printf("'%c'を押しました。\n", pushed_key);
      torch::Tensor orientation = pose.index({Slc(0, 3), Slc(0, 3)});
      if (pushed_key == 'w') {
        torch::Tensor tmp = torch::tensor({step, 0.0f, 0.0f}, torch::kFloat32).view({3, 1}).cuda();
        pose.index({Slc(0, 3), Slc(3, 4)}) += orientation.matmul(tmp);
      } else if (pushed_key == 'a') {
        torch::Tensor tmp = torch::tensor({0.0f, step, 0.0f}, torch::kFloat32).view({3, 1}).cuda();
        pose.index({Slc(0, 3), Slc(3, 4)}) += orientation.matmul(tmp);
      } else if (pushed_key == 'd') {
        torch::Tensor tmp = torch::tensor({0.0f, -step, 0.0f}, torch::kFloat32).view({3, 1}).cuda();
        pose.index({Slc(0, 3), Slc(3, 4)}) += orientation.matmul(tmp);
      } else if (pushed_key == 's') {
        torch::Tensor tmp = torch::tensor({-step, 0.0f, 0.0f}, torch::kFloat32).view({3, 1}).cuda();
        pose.index({Slc(0, 3), Slc(3, 4)}) += orientation.matmul(tmp);
      } else if (pushed_key == 'e') {
        torch::Tensor tmp = torch::tensor({0.0f, 0.0f, step}, torch::kFloat32).view({3, 1}).cuda();
        pose.index({Slc(0, 3), Slc(3, 4)}) += orientation.matmul(tmp);
      } else if (pushed_key == 'q') {
        torch::Tensor tmp = torch::tensor({0.0f, 0.0f, -step}, torch::kFloat32).view({3, 1}).cuda();
        pose.index({Slc(0, 3), Slc(3, 4)}) += orientation.matmul(tmp);
      } else if (pushed_key == 'j') {
        torch::Tensor rotation_matrix = calc_rotation_tensor(-degree, Eigen::Vector3f::UnitZ());
        orientation = rotation_matrix.matmul(orientation);
        pose.index({Slc(0, 3), Slc(0, 3)}) = orientation;
      } else if (pushed_key == 'k') {
        torch::Tensor rotation_matrix = calc_rotation_tensor(-degree, Eigen::Vector3f::UnitY());
        orientation = rotation_matrix.matmul(orientation);
        pose.index({Slc(0, 3), Slc(0, 3)}) = orientation;
      } else if (pushed_key == 'l') {
        torch::Tensor rotation_matrix = calc_rotation_tensor(+degree, Eigen::Vector3f::UnitZ());
        orientation = rotation_matrix.matmul(orientation);
        pose.index({Slc(0, 3), Slc(0, 3)}) = orientation;
      } else if (pushed_key == 'i') {
        torch::Tensor rotation_matrix = calc_rotation_tensor(+degree, Eigen::Vector3f::UnitY());
        orientation = rotation_matrix.matmul(orientation);
        pose.index({Slc(0, 3), Slc(0, 3)}) = orientation;
      } else {
        std::cout << "Unknown kye: " << pushed_key << std::endl;
        continue;
      }
      std::cout << "pose:\n" << pose << std::endl;
      torch::Tensor pose_camera = localizer.world2camera(pose);
      BoundedRays rays = localizer.rays_from_pose(pose_camera);
      const int ray_num = rays.origins.size(0);
      constexpr int kBatchSize = 5000;
      std::vector<torch::Tensor> pred_colors_list;
      for (int i = 0; i < ray_num; i += kBatchSize) {
        const auto [pred_colors, first_oct_disp, pred_disp] = localizer.render_all_rays_grad(
          rays.origins.index({Slc(i, i + kBatchSize)}), rays.dirs.index({Slc(i, i + kBatchSize)}),
          rays.bounds.index({Slc(i, i + kBatchSize)}));
        pred_colors_list.push_back(pred_colors);
      }
      torch::Tensor pred_colors = torch::cat(pred_colors_list, 0);
      pred_colors = pred_colors.clip(0.0f, 1.0f);
      torch::Tensor image =
        pred_colors.view({localizer.infer_height(), localizer.infer_width(), 3});
      Utils::WriteImageTensor("image.png", image);
      std::cout << "WASDで移動, E:上昇, Q下降, J:左回転, K:下回転, L:右回転, I:上回転" << std::endl;
    }
  }
}