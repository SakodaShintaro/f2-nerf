#include "nerf_based_localizer.hpp"

#include "../../src/Utils/Utils.h"

#include <Eigen/Eigen>
#include <experimental/filesystem>
#include <rclcpp/rclcpp.hpp>

#include <torch/torch.h>

#include <sstream>

NerfBasedLocalizer::NerfBasedLocalizer(
  const std::string & name_space, const rclcpp::NodeOptions & options)
: Node("nerf_based_localizer", name_space, options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_),
  map_frame_("map"),
  localizer_core_("./runtime_config.yaml")
{
  this->declare_parameter("save_image", false);

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

  RCLCPP_INFO(this->get_logger(), "nerf_based_localizer is created.");
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

  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_base_link =
    initial_pose_msg_ptr_array_.back();
  initial_pose_msg_ptr_array_.pop_back();
  geometry_msgs::msg::PoseWithCovarianceStamped pose_lidar;
  try {
    geometry_msgs::msg::TransformStamped transform =
      tf_buffer_.lookupTransform("base_link", "lidar", tf2::TimePointZero);
    tf2::doTransform(*pose_base_link, pose_lidar, transform);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
  }

  const geometry_msgs::msg::Pose pose = pose_lidar.pose.pose;

  Eigen::Quaternionf quat(
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  Eigen::Matrix3f rot = quat.toRotationMatrix();

  torch::Tensor initial_pose = torch::eye(4);
  initial_pose[0][0] = rot(0, 0);
  initial_pose[0][1] = rot(0, 1);
  initial_pose[0][2] = rot(0, 2);
  initial_pose[0][3] = pose.position.x;
  initial_pose[1][0] = rot(1, 0);
  initial_pose[1][1] = rot(1, 1);
  initial_pose[1][2] = rot(1, 2);
  initial_pose[1][3] = pose.position.y;
  initial_pose[2][0] = rot(2, 0);
  initial_pose[2][1] = rot(2, 1);
  initial_pose[2][2] = rot(2, 2);
  initial_pose[2][3] = pose.position.z;
  initial_pose = initial_pose.to(torch::kCUDA);
  initial_pose = initial_pose.to(torch::kFloat32);

  ss.str("");
  ss << "initial_pose_fist: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  /*
    [[0, 0, -1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]]
  */
  torch::Tensor axis_convert_mat1 = torch::zeros({4, 4});
  axis_convert_mat1[0][2] = -1;
  axis_convert_mat1[1][0] = -1;
  axis_convert_mat1[2][1] = -1;
  axis_convert_mat1[3][3] = 1;
  axis_convert_mat1 = axis_convert_mat1.to(torch::kCUDA);

  /*
    [[0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]]
  */
  torch::Tensor axis_convert_mat2 = torch::zeros({4, 4});
  axis_convert_mat2[0][1] = -1;
  axis_convert_mat2[1][2] = -1;
  axis_convert_mat2[2][0] = 1;
  axis_convert_mat2[3][3] = 1;
  axis_convert_mat2 = axis_convert_mat2.to(torch::kCUDA);

  /*
   [[ 0.737494  0.021281  0.155938  2.00682 ]
   [ 0.029287 -0.752681 -0.035792 -0.044822]
   [-0.154634 -0.04106   0.736932 -1.760475]
   [ 0.        0.        0.        1.      ]]
  */
  torch::Tensor convert_mat = torch::zeros({4, 4});
  convert_mat[0][0] = 0.737494;
  convert_mat[0][1] = 0.021281;
  convert_mat[0][2] = 0.155938;
  convert_mat[0][3] = 2.00682;
  convert_mat[1][0] = 0.029287;
  convert_mat[1][1] = -0.752681;
  convert_mat[1][2] = -0.035792;
  convert_mat[1][3] = -0.044822;
  convert_mat[2][0] = -0.154634;
  convert_mat[2][1] = -0.04106;
  convert_mat[2][2] = 0.736932;
  convert_mat[2][3] = -1.760475;
  convert_mat[3][3] = 1;
  convert_mat = convert_mat.to(torch::kCUDA);

  initial_pose = initial_pose.matmul(axis_convert_mat1);
  initial_pose = axis_convert_mat2.matmul(initial_pose);
  initial_pose = convert_mat.matmul(initial_pose);

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

  // save image
  if (this->get_parameter("save_image").as_bool()) {
    static int cnt = 0;
    namespace fs = std::experimental::filesystem::v1;
    fs::create_directories("./result_images/trial/pred/");
    fs::create_directories("./result_images/trial/gt/");
    save_image(nerf_image, "./result_images/trial/pred/", cnt);
    save_image(image_tensor, "./result_images/trial/gt/", cnt);
    cnt++;
  }

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

void NerfBasedLocalizer::save_image(
  const torch::Tensor image_tensor, const std::string & prefix, int save_id)
{
  std::stringstream ss;
  ss << prefix;
  ss << std::setfill('0') << std::setw(8) << save_id;
  ss << ".png";
  Utils::WriteImageTensor(ss.str(), image_tensor);
}
