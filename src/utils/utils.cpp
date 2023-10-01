//
// Created by ppwang on 2023/4/4.
//

#include "../Common.h"
#include "utils.hpp"

#include <opencv2/opencv.hpp>

#include <torch/torch.h>

using Tensor = torch::Tensor;

Tensor Utils::ReadImageTensor(const std::string & path)
{
  cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);
  Tensor img_tensor =
    torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, torch::kFloat32).clone();
  return img_tensor;
}

bool Utils::WriteImageTensor(const std::string & path, Tensor img)
{
  img = img.contiguous();
  img = (img * 255.f).clamp(0, 255).to(torch::kUInt8).to(torch::kCPU);
  cv::Mat img_mat(img.size(0), img.size(1), CV_8UC3, img.data_ptr());
  cv::cvtColor(img_mat, img_mat, cv::COLOR_RGB2BGR);
  cv::imwrite(path, img_mat);
  return true;
}
