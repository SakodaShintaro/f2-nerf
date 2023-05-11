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

  // create publishers
  nerf_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("nerf_pose", 10);
  nerf_pose_with_covariance_publisher_ =
    this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "nerf_pose_with_covariance", 10);
  nerf_score_publisher_ = this->create_publisher<std_msgs::msg::Float32>("nerf_score", 10);
  nerf_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("nerf_image", 10);
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
    if (initial_pose_msg_ptr_array_.size() > 1) {
      initial_pose_msg_ptr_array_.pop_front();
    }
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
    return;
  }

  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose =
    initial_pose_msg_ptr_array_.back();
  initial_pose_msg_ptr_array_.pop_back();

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

  ss.str("");
  ss << "initial_pose_fist: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  /*
   [[ 0.7352  0.16    0.0506  2.0536]
   [-0.0435 -0.0378  0.7519  0.3601]
   [-0.1621  0.736   0.0276 -1.7599]]
   [ 0.      0.      0.      1.]]
  */
  torch::Tensor mat_translation = torch::zeros({4, 4});
  mat_translation[0][0] = 0.7352;
  mat_translation[0][1] = 0.16;
  mat_translation[0][2] = 0.0506;
  mat_translation[0][3] = 2.0536;
  mat_translation[1][0] = -0.0435;
  mat_translation[1][1] = -0.0378;
  mat_translation[1][2] = 0.7519;
  mat_translation[1][3] = 0.3601;
  mat_translation[2][0] = -0.1621;
  mat_translation[2][1] = 0.736;
  mat_translation[2][2] = 0.0276;
  mat_translation[2][3] = -1.7599;
  mat_translation[3][3] = 1.0;
  mat_translation = mat_translation.to(torch::kCUDA);
  torch::Tensor translation = torch::mm(mat_translation, initial_pose);

  // Change orientation
  /*
    [0.985575,  0.032919, -0.166004],
    [0.032773, -0.999456, -0.003616],
    [-0.166033, -0.001877, -0.986118]])
  */
  torch::Tensor mat_rotation = torch::eye(3);
  mat_rotation[0][0] = 0.985575;
  mat_rotation[0][1] = 0.032919;
  mat_rotation[0][2] = -0.166004;
  mat_rotation[1][0] = 0.032773;
  mat_rotation[1][1] = -0.999456;
  mat_rotation[1][2] = -0.003616;
  mat_rotation[2][0] = -0.166033;
  mat_rotation[2][1] = -0.001877;
  mat_rotation[2][2] = -0.986118;
  mat_rotation = mat_rotation.to(torch::kCUDA);
  torch::Tensor rotation = torch::mm(mat_rotation, initial_pose.index({Slc(0, 3), Slc(0, 3)}));

  // set initial pose
  initial_pose.index_put_({Slc(0, 3), Slc(0, 3)}, rotation);
  initial_pose.index_put_({Slc(0, 3), 3}, translation.index({Slc(0, 3), 3}));

  // output about pose
  ss.str("");
  ss << "initial_pose_before_norm: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  initial_pose = localizer_core_.normalize_position(initial_pose);

  initial_pose = initial_pose.index({Slc(0, 3), Slc(0, 4)});

  // output about pose
  ss.str("");
  ss << "initial_pose_final: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  // run NeRF
  RCLCPP_INFO(this->get_logger(), "start localize");
  auto [score, optimized_pose, nerf_image] =
    localizer_core_.monte_carlo_localize(initial_pose, image_tensor);

  RCLCPP_INFO(this->get_logger(), ("score = " + std::to_string(score)).c_str());

  // publish image
  nerf_image = nerf_image * 255;
  nerf_image = nerf_image.to(torch::kUInt8);
  nerf_image = nerf_image.to(torch::kCPU);
  nerf_image = nerf_image.contiguous();
  sensor_msgs::msg::Image nerf_image_msg;
  nerf_image_msg.header = header;
  nerf_image_msg.width = nerf_image.size(1);
  nerf_image_msg.height = nerf_image.size(0);
  nerf_image_msg.step = nerf_image.size(1) * 3;
  nerf_image_msg.encoding = "rgb8";
  nerf_image_msg.data.resize(nerf_image.numel());
  std::copy(
    nerf_image.data_ptr<uint8_t>(), nerf_image.data_ptr<uint8_t>() + nerf_image.numel(),
    nerf_image_msg.data.begin());
  nerf_image_publisher_->publish(nerf_image_msg);
}
