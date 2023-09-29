//
// Created by ppwang on 2022/5/7.
//
#include "Dataset.h"
#include <iostream>
#include <fmt/core.h>
#include <experimental/filesystem>
#include "../Utils/cnpy.h"
#include "../Utils/Utils.h"
#include "../Utils/StopWatch.h"

using Tensor = torch::Tensor;

namespace fs = std::experimental::filesystem::v1;

Dataset::Dataset(const YAML::Node & root_config) : config_(root_config)
{
  ScopeWatch dataset_watch("Dataset::Dataset");
  const YAML::Node config = root_config["dataset"];

  const auto data_path = config["data_path"].as<std::string>();
  std::cout << "data_path = " << data_path << std::endl;
  const auto factor = config["factor"].as<float>();

  // Load camera pose
  CHECK(fs::exists(data_path + "/cams_meta.npy"));
  {
    cnpy::NpyArray arr = cnpy::npy_load(data_path + "/cams_meta.npy");
    auto options = torch::TensorOptions().dtype(torch::kFloat64);  // WARN: Float64 Here!!!!!
    Tensor cam_data = torch::from_blob(arr.data<double>(), arr.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);

    n_images_ = arr.shape[0];
    cam_data = cam_data.reshape({n_images_, 27});
    Tensor poses = cam_data.slice(1, 0, 12).reshape({-1, 3, 4}).contiguous();

    Tensor intri = cam_data.slice(1, 12, 21).reshape({-1, 3, 3}).contiguous();
    intri.index_put_({Slc(), Slc(0, 2), Slc(0, 3)}, intri.index({Slc(), Slc(0, 2), Slc(0, 3)}) / factor);

    Tensor dist_params = cam_data.slice(1, 21, 25).reshape({-1, 4}).contiguous();   // [k1, k2, p1, p2]
    Tensor bounds = cam_data.slice(1, 25, 27).reshape({-1, 2}).contiguous();

    poses_ = poses.to(torch::kCUDA).contiguous();
    intri_ = intri.to(torch::kCUDA).contiguous();
    dist_params_ = dist_params.to(torch::kCUDA).contiguous();
    bounds_ = bounds.to(torch::kCUDA).contiguous();
  }

  NormalizeScene();

  // Relax bounds
  auto bounds_factor = config["bounds_factor"].as<std::vector<float>>();
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

  // Load train/test/val split info
  try {
    cnpy::NpyArray sp_arr = cnpy::npy_load(data_path + "/split.npy");
    CHECK_EQ(sp_arr.shape[0], n_images_);

    auto sp_arr_ptr = sp_arr.data<unsigned char>();
    for (int i = 0; i < n_images_; i++) {
      int st = sp_arr_ptr[i];
      split_info_.push_back(st);
      if ((st & 1) == 1) train_set_.push_back(i);
      if ((st & 2) == 2) test_set_.push_back(i);
      if ((st & 4) == 4) val_set_.push_back(i);
    }
  }
  catch (...) {
    for (int i = 0; i < n_images_; i++) {
      if (i % 8 == 0) test_set_.push_back(i);
      else train_set_.push_back(i);
    }
  }
  std::cout << "Number of train/test/val images: " << train_set_.size() << "/" << test_set_.size()
            << "/" << val_set_.size() << std::endl;

  // Prepare training images
  height_ = images[0].size(0);
  width_  = images[0].size(1);
  image_tensors_ = torch::stack(images, 0).contiguous();
}

void Dataset::NormalizeScene() {
  // Given poses_ & bounds_, Gen new poses_, bounds_.
  const auto& config = config_["dataset"];
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
  const std::string base_exp_dir = config_["base_exp_dir"].as<std::string>();
  std::ofstream ofs(base_exp_dir + "/inference_params.yaml");
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

std::tuple<BoundedRays, Tensor, Tensor> Dataset::RandRaysData(int batch_size, int sets) {
  std::vector<int> img_idx;
  if ((sets & DATA_TRAIN_SET) != 0) {
    img_idx.insert(img_idx.end(), train_set_.begin(), train_set_.end());
  }
  if ((sets & DATA_VAL_SET) != 0) {
    img_idx.insert(img_idx.end(), val_set_.begin(), val_set_.end());
  }
  if ((sets & DATA_TEST_SET) != 0) {
    img_idx.insert(img_idx.end(), test_set_.begin(), test_set_.end());
  }
  Tensor cur_set = torch::from_blob(img_idx.data(), { int(img_idx.size())}, CPUInt);
  Tensor cam_indices = torch::randint(int(img_idx.size()), { batch_size }, CPULong); // Torch index need "long long" type
  cam_indices = cur_set.index({cam_indices}).contiguous();
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
