#include "nerf_based_localizer.hpp"

#include "../../src/Utils/Utils.h"
#include "../../src/Utils/StopWatch.h"

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
  tf2_broadcaster_(*this),
  map_frame_("map"),
  is_activated_(false)
{
  this->declare_parameter("save_image", false);
  this->declare_parameter("save_particles", false);
  this->declare_parameter("save_particles_images", false);
  this->declare_parameter("particle_num", 100);
  this->declare_parameter("output_covariance", 0.1);
  this->declare_parameter("base_score", 40.0f);
  target_frame_ = this->declare_parameter<std::string>("target_frame");

  LocalizerCoreParam param;
  param.runtime_config_path = this->declare_parameter<std::string>("runtime_config_path");
  param.render_pixel_num = this->declare_parameter<int>("render_pixel_num");
  param.noise_position_x = this->declare_parameter<float>("noise_position_x");
  param.noise_position_y = this->declare_parameter<float>("noise_position_y");
  param.noise_position_z = this->declare_parameter<float>("noise_position_z");
  param.noise_rotation_x = this->declare_parameter<float>("noise_rotation_x");
  param.noise_rotation_y = this->declare_parameter<float>("noise_rotation_y");
  param.noise_rotation_z = this->declare_parameter<float>("noise_rotation_z");
  param.resize_factor = this->declare_parameter<int>("resize_factor");
  localizer_core_ = LocalizerCore(param);

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

  previous_score_ = this->get_parameter("base_score").as_double();

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
    RCLCPP_ERROR(this->get_logger(), "NerfBasedLocalizer is not activated in callback_image.");
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
  pose_stamped_msg.header.frame_id = map_frame_;
  pose_stamped_msg.header.stamp = image_msg_ptr->header.stamp;
  pose_stamped_msg.pose = pose_msg;
  nerf_pose_publisher_->publish(pose_stamped_msg);

  // (2) publish nerf_pose_with_covariance
  geometry_msgs::msg::PoseWithCovarianceStamped pose_with_cov_msg;
  pose_with_cov_msg.header.frame_id = map_frame_;
  pose_with_cov_msg.header.stamp = image_msg_ptr->header.stamp;
  pose_with_cov_msg.pose.pose = pose_msg;
  const double cov = this->get_parameter("output_covariance").as_double();
  pose_with_cov_msg.pose.covariance[0] = cov;
  pose_with_cov_msg.pose.covariance[7] = cov;
  pose_with_cov_msg.pose.covariance[14] = cov;
  pose_with_cov_msg.pose.covariance[21] = cov * 10;
  pose_with_cov_msg.pose.covariance[28] = cov * 10;
  pose_with_cov_msg.pose.covariance[35] = cov * 10;
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
  RCLCPP_INFO(this->get_logger(), "start NerfBasedLocalizer::service");

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
  res->pose_with_covariance.header.frame_id = map_frame_;
  res->pose_with_covariance.header.stamp = image_msg_ptr->header.stamp;
  res->pose_with_covariance.pose.pose = pose_msg;
  res->pose_with_covariance.pose.covariance = req->pose_with_covariance.pose.covariance;

  RCLCPP_INFO(this->get_logger(), "finish NerfBasedLocalizer::service");
}

std::tuple<geometry_msgs::msg::Pose, sensor_msgs::msg::Image, std_msgs::msg::Float32>
NerfBasedLocalizer::localize(
  const geometry_msgs::msg::Pose & pose_msg, const sensor_msgs::msg::Image & image_msg)
{
  Timer timer;
  timer.start();

  // Get data of image_ptr
  // Accessing header information
  const std_msgs::msg::Header header = image_msg.header;

  // Accessing image properties
  const uint32_t width = image_msg.width;
  const uint32_t height = image_msg.height;
  const uint32_t step = image_msg.step;
  const std::string encoding = image_msg.encoding;

  // output information about image
  RCLCPP_INFO_STREAM(
    this->get_logger(),
    "Image received. width: " << width << ", height: " << height << ", step: " << step);

  // Accessing image data
  torch::Tensor image_tensor = torch::tensor(image_msg.data);
  image_tensor = image_tensor.view({height, width, 3});
  if (true) {
    image_tensor = image_tensor.index({Slc(0, 850)});
  }
  image_tensor = image_tensor.to(torch::kFloat32);
  image_tensor /= 255.0;
  image_tensor = image_tensor.flip(2);  // BGR to RGB
  image_tensor = localizer_core_.resize_image(image_tensor);
  image_tensor = image_tensor.to(torch::kCUDA);

  geometry_msgs::msg::PoseWithCovarianceStamped pose_lidar;
  try {
    geometry_msgs::msg::TransformStamped transform =
      tf_buffer_.lookupTransform(target_frame_, "base_link", tf2::TimePointZero);
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
  RCLCPP_INFO_STREAM(this->get_logger(), "world_before:\n" << initial_pose);

  initial_pose = localizer_core_.world2camera(initial_pose);

  // run NeRF
  const double base_score = this->get_parameter("base_score").as_double();
  const float noise_coeff = (base_score > 0 ? base_score / previous_score_ : 1.0f);
  std::vector<Particle> particles = localizer_core_.random_search(
    initial_pose, image_tensor, this->get_parameter("particle_num").as_int(), noise_coeff);

  if (this->get_parameter("save_particles_images").as_bool()) {
    std::sort(particles.begin() + 1, particles.end(), [](const Particle & a, const Particle & b) {
      return a.weight > b.weight;
    });
    static int cnt = 0;
    namespace fs = std::experimental::filesystem::v1;
    fs::create_directories("./result_images/trial/particles_images/");

    for (int32_t i = 0; i < particles.size(); i++) {
      auto [score, nerf_image] =
        localizer_core_.pred_image_and_calc_score(particles[i].pose, image_tensor);
      std::stringstream ss;
      ss << "./result_images/trial/particles_images/";
      save_image(nerf_image, ss.str(), i);
      particles[i].weight = score;
    }
    cnt++;
  }

  torch::Tensor optimized_pose = LocalizerCore::calc_average_pose(particles);

  auto [score, nerf_image] =
    localizer_core_.pred_image_and_calc_score(optimized_pose, image_tensor);

  RCLCPP_INFO_STREAM(this->get_logger(), "score = " << score);
  previous_score_ = score;
  if (this->get_parameter("save_particles").as_bool()) {
    static int cnt = 0;
    namespace fs = std::experimental::filesystem::v1;
    fs::create_directories("./result_images/trial/particles/");
    std::stringstream ss;
    ss << std::setw(8) << std::setfill('0') << cnt;
    std::ofstream ofs("./result_images/trial/particles/" + ss.str() + ".tsv");
    ofs << "m00\tm01\tm02\tm03\tm10\tm11\tm12\tm13\tm20\tm21\tm22\tm23\tweight" << std::endl;
    ofs << std::fixed;
    for (int p = 0; p < particles.size(); p++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
          ofs << particles[p].pose[i][j].item<float>() << "\t";
        }
      }
      ofs << particles[p].weight << std::endl;
    }
    cnt++;
  }

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
  optimized_pose = localizer_core_.camera2world(optimized_pose);

  RCLCPP_INFO_STREAM(this->get_logger(), "world_after:\n" << optimized_pose);

  geometry_msgs::msg::Pose result_pose_lidar;
  result_pose_lidar.position.x = optimized_pose[0][3].item<float>();
  result_pose_lidar.position.y = optimized_pose[1][3].item<float>();
  result_pose_lidar.position.z = optimized_pose[2][3].item<float>();
  Eigen::Matrix3f rot_out;
  rot_out << optimized_pose[0][0].item<float>(), optimized_pose[0][1].item<float>(),
    optimized_pose[0][2].item<float>(), optimized_pose[1][0].item<float>(),
    optimized_pose[1][1].item<float>(), optimized_pose[1][2].item<float>(),
    optimized_pose[2][0].item<float>(), optimized_pose[2][1].item<float>(),
    optimized_pose[2][2].item<float>();
  Eigen::Quaternionf quat_out(rot_out);
  result_pose_lidar.orientation.x = quat_out.x();
  result_pose_lidar.orientation.y = quat_out.y();
  result_pose_lidar.orientation.z = quat_out.z();
  result_pose_lidar.orientation.w = quat_out.w();

  geometry_msgs::msg::Pose result_pose_base_link;
  try {
    geometry_msgs::msg::TransformStamped transform =
      tf_buffer_.lookupTransform("base_link", target_frame_, tf2::TimePointZero);
    tf2::doTransform(result_pose_lidar, result_pose_base_link, transform);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
  }

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

  geometry_msgs::msg::TransformStamped transform;
  transform.transform.translation.x = result_pose_base_link.position.x;
  transform.transform.translation.y = result_pose_base_link.position.y;
  transform.transform.translation.z = result_pose_base_link.position.z;
  transform.transform.rotation = result_pose_base_link.orientation;
  transform.header = header;
  transform.header.frame_id = "map";
  transform.child_frame_id = "nerf_base_link";
  tf2_broadcaster_.sendTransform(transform);

  RCLCPP_INFO_STREAM(get_logger(), "localize time: " << timer.elapsed_milli_seconds());

  return std::make_tuple(result_pose_base_link, nerf_image_msg, score_msg);
}

void NerfBasedLocalizer::service_trigger_node(
  const std_srvs::srv::SetBool::Request::SharedPtr req,
  std_srvs::srv::SetBool::Response::SharedPtr res)
{
  RCLCPP_INFO_STREAM(
    this->get_logger(), "service_trigger " << req->data << " is arrived to NerfBasedLocalizer.");

  is_activated_ = req->data;
  if (is_activated_) {
    std::lock_guard<std::mutex> initial_pose_array_lock(initial_pose_array_mtx_);
    initial_pose_msg_ptr_array_.clear();
    std::lock_guard<std::mutex> image_array_lock(image_array_mtx_);
    image_msg_ptr_array_.clear();
  }
  res->success = true;
}
