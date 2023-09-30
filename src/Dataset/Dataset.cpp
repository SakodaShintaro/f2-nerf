//
// Created by ppwang on 2022/5/7.
//
#include "Dataset.h"
#include <iostream>
#include <fmt/core.h>
#include <experimental/filesystem>
#include "../Utils/Utils.h"
#include "../Utils/StopWatch.h"

using Tensor = torch::Tensor;

namespace fs = std::experimental::filesystem::v1;

Dataset::Dataset(const std::string & data_path, const std::string & output_dir) : output_dir_(output_dir)
{
  ScopeWatch dataset_watch("Dataset::Dataset");

  std::cout << "data_path = " << data_path << std::endl;

  // Load camera pose
  CHECK(fs::exists(data_path + "/cams_meta.tsv"));
  {
    std::ifstream ifs(data_path + "/cams_meta.tsv");
    std::string line;
    std::getline(ifs, line);  // header
    std::vector<Tensor> poses, intrinsics, dist_params, bounds;
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (std::getline(iss, token, '\t')) {
        tokens.push_back(token);
      }
      const int POSE_NUM = 12;       //(3, 4)
      const int INTRINSIC_NUM = 9;   //(3, 3)
      const int DISTORTION_NUM = 4;  //(k1, k2, p1, p2)
      const int BOUNDS_NUM = 2;      //(near, far)
      CHECK_EQ(tokens.size(), POSE_NUM + INTRINSIC_NUM + DISTORTION_NUM + BOUNDS_NUM);
      Tensor pose = torch::zeros({3, 4}, torch::kFloat32);
      for (int i = 0; i < POSE_NUM; i++) {
        pose.index_put_({i / 4, i % 4}, std::stof(tokens[i]));
      }
      pose = pose.reshape({3, 4});
      poses.push_back(pose);

      Tensor intrinsic = torch::zeros({3, 3}, torch::kFloat32);
      for (int i = 0; i < INTRINSIC_NUM; i++) {
        intrinsic.index_put_({i / 3, i % 3}, std::stof(tokens[POSE_NUM + i]));
      }
      intrinsic = intrinsic.reshape({3, 3});
      intrinsics.push_back(intrinsic);

      Tensor dist_param = torch::zeros({4}, torch::kFloat32);
      for (int i = 0; i < DISTORTION_NUM; i++) {
        dist_param.index_put_({i}, std::stof(tokens[POSE_NUM + INTRINSIC_NUM + i]));
      }
      dist_params.push_back(dist_param);

      Tensor bound = torch::zeros({2}, torch::kFloat32);
      for (int i = 0; i < BOUNDS_NUM; i++) {
        bound.index_put_({i}, std::stof(tokens[POSE_NUM + INTRINSIC_NUM + DISTORTION_NUM + i]));
      }
      bounds.push_back(bound);
    }

    n_images_ = poses.size();
    poses_ = torch::stack(poses, 0).contiguous().to(torch::kCUDA);
    intri_ = torch::stack(intrinsics, 0).contiguous().to(torch::kCUDA);
    dist_params_ = torch::stack(dist_params, 0).contiguous().to(torch::kCUDA);
    bounds_ = torch::stack(bounds, 0).contiguous().to(torch::kCUDA);
  }

  NormalizeScene();

  // Relax bounds
  const std::vector<float> bounds_factor = {0.25, 4.0};
  bounds_ = torch::stack( { bounds_.index({"...", 0}) * bounds_factor[0],
                                 { bounds_.index({"...", 1}) * bounds_factor[1]}}, -1).contiguous();
  bounds_.clamp_(1e-2f, 1e9f);

  std::vector<Tensor> images;
  // Load images
  {
    ScopeWatch watch("LoadImages");
    std::ifstream image_list(data_path + "/image_list.txt");
    for (int i = 0; i < n_images_; i++) {
      std::string image_path;
      std::getline(image_list, image_path);
      images.push_back(Utils::ReadImageTensor(image_path).to(torch::kCPU));
    }
  }

  std::cout << "Number of images: " << n_images_ << std::endl;

  // Prepare training images
  height_ = images[0].size(0);
  width_  = images[0].size(1);
  image_tensors_ = torch::stack(images, 0).contiguous();
}

void Dataset::NormalizeScene() {
  // Given poses_ & bounds_, Gen new poses_, bounds_.
  Tensor cam_pos = poses_.index({Slc(), Slc(0, 3), 3}).clone();
  center_ = cam_pos.mean(0, false);
  Tensor bias = cam_pos - center_.unsqueeze(0);
  radius_ = torch::linalg_norm(bias, 2, -1, false).max().item<float>();
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  poses_.index_put_({Slc(), Slc(0, 3), 3}, cam_pos);
  poses_ = poses_.contiguous();
  bounds_ = (bounds_ / radius_).contiguous();
}

void Dataset::SaveInferenceParams() const
{
  std::ofstream ofs(output_dir_ + "/inference_params.yaml");
  ofs << std::fixed;
  ofs << "n_images: " << n_images_ << std::endl;
  ofs << "height: " << height_ << std::endl;
  ofs << "width: " << width_ << std::endl;

  ofs << "intrinsic: [";
  ofs << intri_[0][0][0].item() << ", ";
  ofs << intri_[0][0][1].item() << ", ";
  ofs << intri_[0][0][2].item() << "," << std::endl;
  ofs << "            ";
  ofs << intri_[0][1][0].item() << ", ";
  ofs << intri_[0][1][1].item() << ", ";
  ofs << intri_[0][1][2].item() << "," << std::endl;
  ofs << "            ";
  ofs << intri_[0][2][0].item() << ", ";
  ofs << intri_[0][2][1].item() << ", ";
  ofs << intri_[0][2][2].item() << "]" << std::endl;

  ofs << "bounds: [" << bounds_[0][0].item();
  ofs << ", " << bounds_[0][1].item() << "]" << std::endl;

  ofs << "normalizing_center: [" << center_[0].item();
  ofs << ", " << center_[1].item();
  ofs << ", " << center_[2].item() << "]" << std::endl;
  ofs << "normalizing_radius: " << radius_ << std::endl;
}

Rays Dataset::Img2WorldRay(const Tensor& pose,
                           const Tensor& intri,
                           const Tensor& ij) {
  Tensor i = ij.index({"...", 0}).to(torch::kFloat32) + .5f;
  Tensor j = ij.index({"...", 1}).to(torch::kFloat32) + .5f; // Shift half pixel;

  Tensor cx = intri.index({Slc(), 0, 2});
  Tensor cy = intri.index({Slc(), 1, 2});
  Tensor fx = intri.index({Slc(), 0, 0});
  Tensor fy = intri.index({Slc(), 1, 1});

  Tensor u_tensor = ((j - cx) / fx).unsqueeze(-1);
  Tensor v_tensor = -((i - cy) / fy).unsqueeze(-1);
  Tensor w_tensor = -torch::ones_like(u_tensor);

  Tensor dir_tensor = torch::cat({u_tensor, v_tensor, w_tensor}, 1).unsqueeze(-1);
  Tensor ori_tensor = pose.index({Slc(), Slc(0, 3), Slc(0, 3)});
  Tensor pos_tensor = pose.index({Slc(), Slc(0, 3), 3});
  Tensor rays_d = torch::matmul(ori_tensor, dir_tensor).squeeze();
  Tensor rays_o = pos_tensor.expand({ rays_d.sizes()[0], 3 }).contiguous();

  return { rays_o, rays_d };
}

BoundedRays Dataset::RaysOfCamera(int idx, int reso_level) {
  int H = height_;
  int W = width_;
  Tensor ii = torch::linspace(0.f, H - 1.f, H, CUDAFloat);
  Tensor jj = torch::linspace(0.f, W - 1.f, W, CUDAFloat);
  auto ij = torch::meshgrid({ ii, jj }, "ij");
  Tensor i = ij[0].reshape({-1});
  Tensor j = ij[1].reshape({-1});

  float near = bounds_.index({idx, 0}).item<float>();
  float far  = bounds_.index({idx, 1}).item<float>();

  Tensor bounds = torch::stack({
                                   torch::full({ H * W }, near, CUDAFloat),
                                   torch::full({ H * W }, far,  CUDAFloat)
                               }, -1).contiguous();

  auto [rays_o, rays_d] =
    Img2WorldRay(poses_[idx].unsqueeze(0), intri_[idx].unsqueeze(0), torch::stack({i, j}, -1));
  return { rays_o, rays_d, bounds };
}

std::tuple<BoundedRays, Tensor, Tensor> Dataset::RandRaysData(int batch_size) {
  Tensor cam_indices = torch::randint(n_images_, {batch_size}, CPULong);
  Tensor i = torch::randint(0, height_, batch_size, CPULong);
  Tensor j = torch::randint(0, width_, batch_size, CPULong);
  Tensor ij = torch::stack({i, j}, -1).to(torch::kCUDA).contiguous();

  Tensor gt_colors = image_tensors_.view({-1, 3})
                       .index({(cam_indices * height_ * width_ + i * width_ + j).to(torch::kLong)})
                       .to(torch::kCUDA)
                       .contiguous();
  cam_indices = cam_indices.to(torch::kCUDA);
  cam_indices = cam_indices.to(torch::kInt32);
  ij = ij.to(torch::kInt32);

  Tensor selected_poses = torch::index_select(poses_, 0, cam_indices);
  Tensor selected_intri = torch::index_select(intri_, 0, cam_indices);
  auto [rays_o, rays_d] = Img2WorldRay(selected_poses, selected_intri, ij);

  Tensor bounds = bounds_.index({cam_indices.to(torch::kLong)}).contiguous();
  return {{rays_o, rays_d, bounds}, gt_colors, cam_indices.to(torch::kInt32).contiguous()};
}
