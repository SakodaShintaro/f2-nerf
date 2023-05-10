#include "nerf_based_localizer.hpp"

#include <Eigen/Eigen>
#include <rclcpp/rclcpp.hpp>

#include <torch/torch.h>

#include <sstream>

NerfBasedLocalizer::NerfBasedLocalizer(
  const std::string & name_space, const rclcpp::NodeOptions & options)
: Node("nerf_based_localizer", name_space, options),
  map_frame_("map"),
  localizer_core_("./runtime_config.yaml")
{
  RCLCPP_INFO(this->get_logger(), "nerf_based_localizer is created.");

  initial_pose_with_covariance_subscriber_ =
    this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "initial_pose_with_covariance", 100,
      std::bind(&NerfBasedLocalizer::callback_initial_pose, this, std::placeholders::_1));

  int image_queue_size = this->declare_parameter("input_sensor_points_queue_size", 0);
  image_queue_size = std::max(image_queue_size, 0);
  image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
    "image", rclcpp::SensorDataQoS().keep_last(image_queue_size),
    std::bind(&NerfBasedLocalizer::callback_image, this, std::placeholders::_1));
}

void NerfBasedLocalizer::callback_initial_pose(
  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr initial_pose_msg_ptr)
{
  // lock mutex for initial pose
  std::lock_guard<std::mutex> initial_pose_array_lock(initial_pose_array_mtx_);
  // if rosbag restart, clear buffer
  if (!initial_pose_msg_ptr_array_.empty()) {
    const builtin_interfaces::msg::Time & t_front =
      initial_pose_msg_ptr_array_.front()->header.stamp;
    const builtin_interfaces::msg::Time & t_msg = initial_pose_msg_ptr->header.stamp;
    if (t_front.sec > t_msg.sec || (t_front.sec == t_msg.sec && t_front.nanosec > t_msg.nanosec)) {
      initial_pose_msg_ptr_array_.clear();
    }
  }

  if (initial_pose_msg_ptr->header.frame_id == map_frame_) {
    initial_pose_msg_ptr_array_.push_back(initial_pose_msg_ptr);
  } else {
    RCLCPP_ERROR(this->get_logger(), "initial_pose_with_covariance is not in map frame.");
    std::exit(1);
  }
}

void NerfBasedLocalizer::callback_image(const sensor_msgs::msg::Image::ConstSharedPtr image_msg_ptr)
{
  // Get data of image_ptr
  // Accessing header information
  std_msgs::msg::Header header = image_msg_ptr->header;

  // Accessing image properties
  uint32_t width = image_msg_ptr->width;
  uint32_t height = image_msg_ptr->height;
  uint32_t step = image_msg_ptr->step;
  std::string encoding = image_msg_ptr->encoding;

  // output information about image
  std::stringstream ss;
  ss << "Image received. ";
  ss << "width: " << width << ", ";
  ss << "height: " << height << ", ";
  ss << "step: " << step;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  // Accessing image data
  torch::Tensor image_tensor = torch::tensor(image_msg_ptr->data);
  image_tensor = image_tensor.view({height, width, 3});
  image_tensor = image_tensor.to(torch::kCUDA);
  image_tensor = image_tensor.to(torch::kFloat32);
  image_tensor /= 255.0;

  // lock mutex for initial pose
  std::lock_guard<std::mutex> initial_pose_array_lock(initial_pose_array_mtx_);

  if (initial_pose_msg_ptr_array_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "initial_pose_with_covariance is not received.");
    std::exit(1);
  }

  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose =
    initial_pose_msg_ptr_array_.front();
  initial_pose_msg_ptr_array_.pop_front();

  Eigen::Quaternionf quat(
    pose->pose.pose.orientation.w, pose->pose.pose.orientation.x, pose->pose.pose.orientation.y,
    pose->pose.pose.orientation.z);
  Eigen::Matrix3f rot = quat.toRotationMatrix();

  torch::Tensor initial_pose = torch::eye(4);
  initial_pose[0][0] = rot(0, 0);
  initial_pose[0][1] = rot(0, 1);
  initial_pose[0][2] = rot(0, 2);
  initial_pose[0][3] = pose->pose.pose.position.x;
  initial_pose[1][0] = rot(1, 0);
  initial_pose[1][1] = rot(1, 1);
  initial_pose[1][2] = rot(1, 2);
  initial_pose[1][3] = pose->pose.pose.position.y;
  initial_pose[2][0] = rot(2, 0);
  initial_pose[2][1] = rot(2, 1);
  initial_pose[2][2] = rot(2, 2);
  initial_pose[2][3] = pose->pose.pose.position.z;
  initial_pose = initial_pose.to(torch::kCUDA);
  initial_pose = initial_pose.to(torch::kFloat32);

  // torch::Tensor mat = torch::zeros({4, 4});
  // mat[0][0] = 0.001807;
  // mat[0][1] = -0.750991;
  // mat[0][2] = -0.009577;
  // mat[0][3] = 2.456662;
  // mat[1][0] = 0.026115;
  // mat[1][1] = 0.009634;
  // mat[1][2] = -0.750538;
  // mat[1][3] = 0.161284;
  // mat[2][0] = 0.750598;
  // mat[2][1] = 0.001472;
  // mat[2][2] = 0.026136;
  // mat[2][3] = -1.306355;
  // mat[3][3] = 1.0;
  // mat = mat.to(torch::kCUDA);
  // initial_pose = mat * initial_pose;

  initial_pose = localizer_core_.normalize_position(initial_pose);

  initial_pose = initial_pose.index({Slc(0, 3), Slc(0, 4)});

  // output about pose
  ss.str("");
  ss << "initial_pose: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  // run NeRF
  RCLCPP_INFO(this->get_logger(), "start localize");
  const auto [score, optimized_pose] =
    localizer_core_.monte_carlo_localize(initial_pose, image_tensor);

  RCLCPP_INFO(this->get_logger(), ("score = " + std::to_string(score)).c_str());
}
