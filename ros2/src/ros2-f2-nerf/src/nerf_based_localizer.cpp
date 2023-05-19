#include "nerf_based_localizer.hpp"

#include "../../src/Utils/Utils.h"
#include "timer.hpp"

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
  is_activated_(false),
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

  /*
    [[0, 0, -1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]]
  */
  axis_convert_mat1_ = torch::zeros({4, 4});
  axis_convert_mat1_[0][2] = -1;
  axis_convert_mat1_[1][0] = -1;
  axis_convert_mat1_[2][1] = -1;
  axis_convert_mat1_[3][3] = 1;
  axis_convert_mat1_ = axis_convert_mat1_.to(torch::kCUDA);

  /*
    [[0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]]
  */
  axis_convert_mat2_ = torch::zeros({4, 4});
  axis_convert_mat2_[0][1] = -1;
  axis_convert_mat2_[1][2] = -1;
  axis_convert_mat2_[2][0] = 1;
  axis_convert_mat2_[3][3] = 1;
  axis_convert_mat2_ = axis_convert_mat2_.to(torch::kCUDA);

  /*
    [[ 1.327284  0.008350  0.118095  1.719275]
     [-0.011544  1.332029  0.035553  3.282300]
     [ 0.117825  0.036435 -1.326834  0.158593]
     [ 0.        0.        0.        1.      ]]
  */
  convert_mat_A2B_ = torch::zeros({4, 4});
  convert_mat_A2B_[0][0] = 1.327284;
  convert_mat_A2B_[0][1] = 0.008350;
  convert_mat_A2B_[0][2] = 0.118095;
  convert_mat_A2B_[0][3] = 1.719275;
  convert_mat_A2B_[1][0] = -0.011544;
  convert_mat_A2B_[1][1] = 1.332029;
  convert_mat_A2B_[1][2] = 0.035553;
  convert_mat_A2B_[1][3] = 3.282300;
  convert_mat_A2B_[2][0] = 0.117825;
  convert_mat_A2B_[2][1] = 0.036435;
  convert_mat_A2B_[2][2] = -1.326834;
  convert_mat_A2B_[2][3] = 0.158593;
  convert_mat_A2B_[3][3] = 1;
  convert_mat_A2B_ = convert_mat_A2B_.to(torch::kCUDA);

  /*
    [[ 0.750144  0.020519 -0.004703  2.473535]
     [ 0.020022 -0.747218 -0.066506  0.061556]
     [ 0.006501 -0.066354  0.747471 -1.274295]
     [ 0.        0.        0.        1.      ]]
  */
  convert_mat_B2A_ = torch::zeros({4, 4});
  convert_mat_B2A_[0][0] = 0.750144;
  convert_mat_B2A_[0][1] = 0.020519;
  convert_mat_B2A_[0][2] = -0.004703;
  convert_mat_B2A_[0][3] = 2.473535;
  convert_mat_B2A_[1][0] = 0.020022;
  convert_mat_B2A_[1][1] = -0.747218;
  convert_mat_B2A_[1][2] = -0.066506;
  convert_mat_B2A_[1][3] = 0.061556;
  convert_mat_B2A_[2][0] = 0.006501;
  convert_mat_B2A_[2][1] = -0.066354;
  convert_mat_B2A_[2][2] = 0.747471;
  convert_mat_B2A_[2][3] = -1.274295;
  convert_mat_B2A_[3][3] = 1;
  convert_mat_B2A_ = convert_mat_B2A_.to(torch::kCUDA);

  service_ = this->create_service<tier4_localization_msgs::srv::PoseWithCovarianceStamped>(
    "nerf_service",
    std::bind(&NerfBasedLocalizer::service, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

  service_trigger_node_ = this->create_service<std_srvs::srv::SetBool>(
    "trigger_node_srv",
    std::bind(
      &NerfBasedLocalizer::service_trigger_node, this, std::placeholders::_1,
      std::placeholders::_2),
    rclcpp::ServicesQoS().get_rmw_qos_profile());

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
  // lock mutex for image
  std::lock_guard<std::mutex> image_array_lock(image_array_mtx_);
  image_msg_ptr_array_.push_back(image_msg_ptr);
  if (image_msg_ptr_array_.size() > 1) {
    image_msg_ptr_array_.pop_front();
  }

  if (!is_activated_) {
    return;
  }

  // lock mutex for initial pose
  std::lock_guard<std::mutex> initial_pose_array_lock(initial_pose_array_mtx_);
  if (initial_pose_msg_ptr_array_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "initial_pose_with_covariance is not received.");
    return;
  }

  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_base_link =
    initial_pose_msg_ptr_array_.back();
  initial_pose_msg_ptr_array_.pop_back();

  // Process
  const auto [pose_msg, image_msg, score_msg] = localize(pose_base_link->pose.pose, *image_msg_ptr);

  // (1) publish nerf_pose
  geometry_msgs::msg::PoseStamped pose_stamped_msg;
  pose_stamped_msg.header = pose_base_link->header;
  pose_stamped_msg.pose = pose_msg;
  nerf_pose_publisher_->publish(pose_stamped_msg);

  // (2) publish nerf_pose_with_covariance
  geometry_msgs::msg::PoseWithCovarianceStamped pose_with_cov_msg;
  pose_with_cov_msg.header = pose_base_link->header;
  pose_with_cov_msg.pose.pose = pose_msg;
  pose_with_cov_msg.pose.covariance[0] = 0.1;
  pose_with_cov_msg.pose.covariance[7] = 0.1;
  pose_with_cov_msg.pose.covariance[14] = 0.1;
  pose_with_cov_msg.pose.covariance[21] = 0.1;
  pose_with_cov_msg.pose.covariance[28] = 0.1;
  pose_with_cov_msg.pose.covariance[35] = 0.1;
  nerf_pose_with_covariance_publisher_->publish(pose_with_cov_msg);

  // (3) publish score
  nerf_score_publisher_->publish(score_msg);

  // (4) publish image
  nerf_image_publisher_->publish(image_msg);
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

void NerfBasedLocalizer::service(
  const tier4_localization_msgs::srv::PoseWithCovarianceStamped::Request::SharedPtr req,
  tier4_localization_msgs::srv::PoseWithCovarianceStamped::Response::SharedPtr res)
{
  // lock mutex for image
  std::lock_guard<std::mutex> image_array_lock(image_array_mtx_);
  if (image_msg_ptr_array_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "image is not received.");
    res->success = false;
    return;
  }

  // Get the oldest image
  const sensor_msgs::msg::Image::ConstSharedPtr image_msg_ptr = image_msg_ptr_array_.back();

  // Process
  const auto [pose_msg, image_msg, score_msg] =
    localize(req->pose_with_covariance.pose.pose, *image_msg_ptr);

  res->success = true;
  res->pose_with_covariance.header = req->pose_with_covariance.header;
  res->pose_with_covariance.pose.pose = pose_msg;
  res->pose_with_covariance.pose.covariance = req->pose_with_covariance.pose.covariance;
}

std::tuple<geometry_msgs::msg::Pose, sensor_msgs::msg::Image, std_msgs::msg::Float32>
NerfBasedLocalizer::localize(
  const geometry_msgs::msg::Pose & pose_msg, const sensor_msgs::msg::Image & image_msg)
{
  Timer timer;

  // Get data of image_ptr
  // Accessing header information
  const std_msgs::msg::Header header = image_msg.header;

  // Accessing image properties
  const uint32_t width = image_msg.width;
  const uint32_t height = image_msg.height;
  const uint32_t step = image_msg.step;
  const std::string encoding = image_msg.encoding;

  // output information about image
  std::stringstream ss;
  ss << "Image received. ";
  ss << "width: " << width << ", ";
  ss << "height: " << height << ", ";
  ss << "step: " << step;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  // Accessing image data
  torch::Tensor image_tensor = torch::tensor(image_msg.data);
  image_tensor = image_tensor.view({height, width, 3});
  image_tensor = image_tensor.to(torch::kCUDA);
  image_tensor = image_tensor.to(torch::kFloat32);
  image_tensor /= 255.0;

  geometry_msgs::msg::PoseWithCovarianceStamped pose_lidar;
  try {
    geometry_msgs::msg::TransformStamped transform =
      tf_buffer_.lookupTransform("velodyne_front", "base_link", tf2::TimePointZero);
    tf2::doTransform(pose_msg, pose_lidar.pose.pose, transform);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
  }

  const geometry_msgs::msg::Pose pose = pose_lidar.pose.pose;

  Eigen::Quaternionf quat_in(
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  Eigen::Matrix3f rot_in = quat_in.toRotationMatrix();

  torch::Tensor initial_pose = torch::eye(4);
  initial_pose[0][0] = rot_in(0, 0);
  initial_pose[0][1] = rot_in(0, 1);
  initial_pose[0][2] = rot_in(0, 2);
  initial_pose[0][3] = pose.position.x;
  initial_pose[1][0] = rot_in(1, 0);
  initial_pose[1][1] = rot_in(1, 1);
  initial_pose[1][2] = rot_in(1, 2);
  initial_pose[1][3] = pose.position.y;
  initial_pose[2][0] = rot_in(2, 0);
  initial_pose[2][1] = rot_in(2, 1);
  initial_pose[2][2] = rot_in(2, 2);
  initial_pose[2][3] = pose.position.z;
  initial_pose = initial_pose.to(torch::kCUDA);
  initial_pose = initial_pose.to(torch::kFloat32);

  ss.str("");
  ss << "initial_pose_fist: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  initial_pose = initial_pose.matmul(axis_convert_mat1_);
  initial_pose = axis_convert_mat2_.matmul(initial_pose);
  initial_pose = convert_mat_B2A_.matmul(initial_pose);
  initial_pose = localizer_core_.normalize_position(initial_pose);
  initial_pose = initial_pose.index({Slc(0, 3), Slc(0, 4)});

  // output about pose
  ss.str("");
  ss << "initial_pose_converted: " << initial_pose;
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());

  // run NeRF
  RCLCPP_INFO(this->get_logger(), "start localize");
  auto [score, nerf_image] = localizer_core_.pred_image_and_calc_score(initial_pose, image_tensor);
  torch::Tensor optimized_pose = initial_pose;

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

  // Convert pose to base_link
  optimized_pose =
    torch::cat({optimized_pose, torch::tensor({0, 0, 0, 1}).view({1, 4}).to(torch::kCUDA)});
  optimized_pose = localizer_core_.inverse_normalize_position(optimized_pose);
  optimized_pose = optimized_pose.matmul(axis_convert_mat1_.t());
  optimized_pose = axis_convert_mat2_.t().matmul(optimized_pose);
  optimized_pose = convert_mat_A2B_.matmul(optimized_pose);

  geometry_msgs::msg::Pose result_pose;
  result_pose.position.x = optimized_pose[0][3].item<float>();
  result_pose.position.y = optimized_pose[1][3].item<float>();
  result_pose.position.z = optimized_pose[2][3].item<float>();
  Eigen::Matrix3f rot_out;
  rot_out << optimized_pose[0][0].item<float>(), optimized_pose[0][1].item<float>(),
    optimized_pose[0][2].item<float>(), optimized_pose[1][0].item<float>(),
    optimized_pose[1][1].item<float>(), optimized_pose[1][2].item<float>(),
    optimized_pose[2][0].item<float>(), optimized_pose[2][1].item<float>(),
    optimized_pose[2][2].item<float>();
  Eigen::Quaternionf quat_out(rot_out);
  result_pose.orientation.x = quat_out.x();
  result_pose.orientation.y = quat_out.y();
  result_pose.orientation.z = quat_out.z();
  result_pose.orientation.w = quat_out.w();

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

  std_msgs::msg::Float32 score_msg;
  score_msg.data = score;

  RCLCPP_INFO_STREAM(get_logger(), "localize time: " << timer);

  return std::make_tuple(result_pose, nerf_image_msg, score_msg);
}

void NerfBasedLocalizer::service_trigger_node(
  const std_srvs::srv::SetBool::Request::SharedPtr req,
  std_srvs::srv::SetBool::Response::SharedPtr res)
{
  RCLCPP_INFO(
    this->get_logger(),
    ("service_trigger " + std::to_string(req->data) + " is arrived to NerfBasedLocalizer.")
      .c_str());

  is_activated_ = req->data;
  if (is_activated_) {
    std::lock_guard<std::mutex> initial_pose_array_lock(initial_pose_array_mtx_);
    initial_pose_msg_ptr_array_.clear();
    std::lock_guard<std::mutex> image_array_lock(image_array_mtx_);
    image_msg_ptr_array_.clear();
  }
  res->success = true;
}
