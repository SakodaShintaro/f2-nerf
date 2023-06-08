#ifndef NERF_BASED_LOCALIZER_HPP_
#define NERF_BASED_LOCALIZER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tier4_localization_msgs/srv/pose_with_covariance_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <deque>

// If you include this first, you will get an error.
// clang-format off
#include "localizer_core.hpp"
// clang-format on

class NerfBasedLocalizer : public rclcpp::Node
{
public:
  NerfBasedLocalizer(
    const std::string & name_space = "",
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void callback_initial_pose(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_conv_msg_ptr);
  void callback_image(const sensor_msgs::msg::Image::ConstSharedPtr image_msg_ptr);
  void service(
    const tier4_localization_msgs::srv::PoseWithCovarianceStamped::Request::SharedPtr req,
    tier4_localization_msgs::srv::PoseWithCovarianceStamped::Response::SharedPtr res);
  void service_trigger_node(
    const std_srvs::srv::SetBool::Request::SharedPtr req,
    std_srvs::srv::SetBool::Response::SharedPtr res);

  std::tuple<geometry_msgs::msg::Pose, sensor_msgs::msg::Image, std_msgs::msg::Float32> localize(
    const geometry_msgs::msg::Pose & pose_msg, const sensor_msgs::msg::Image & image_msg);

  void save_image(const torch::Tensor image_tensor, const std::string & prefix, int save_id);
  torch::Tensor world2camera(const torch::Tensor & pose_in_world);
  torch::Tensor camera2world(const torch::Tensor & pose_in_camera);

  // NerfBasedLocalizer subscribes to the following topics:
  // (1) initial_pose_with_covariance [geometry_msgs::msg::PoseWithCovarianceStamped]
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    initial_pose_with_covariance_subscriber_;
  // (2) image [sensor_msgs::msg::Image]
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;

  // NerfBasedLocalizer publishes the following topics:
  // (1) nerf_pose [geometry_msgs::msg::PoseStamped]
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr nerf_pose_publisher_;
  // (2) nerf_pose_with_covariance [geometry_msgs::msg::PoseWithCovarianceStamped]
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    nerf_pose_with_covariance_publisher_;
  // (3) nerf_score [std_msgs::msg::Float32]
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr nerf_score_publisher_;
  // (4) nerf_image [sensor_msgs::msg::Image]
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr nerf_image_publisher_;

  // tf
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf2_broadcaster_;

  // Convert mat
  torch::Tensor axis_convert_mat1_;

  float previous_score_;

  // Service
  rclcpp::Service<tier4_localization_msgs::srv::PoseWithCovarianceStamped>::SharedPtr service_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_trigger_node_;

  std::string map_frame_;
  std::string target_frame_;

  // data deque
  std::deque<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>
    initial_pose_msg_ptr_array_;
  std::mutex initial_pose_array_mtx_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> image_msg_ptr_array_;
  std::mutex image_array_mtx_;

  bool is_activated_;

  bool is_awsim_;

  LocalizerCore localizer_core_;
};

#endif  // NERF_BASED_LOCALIZER_HPP_
