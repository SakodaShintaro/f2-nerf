#ifndef NERF_BASED_LOCALIZER_HPP_
#define NERF_BASED_LOCALIZER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <tf2_ros/buffer.h>
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
  void publish_pose();

  void save_image(const torch::Tensor image_tensor, const std::string & prefix, int save_id);

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

  // Convert mat
  torch::Tensor axis_convert_mat1_;
  torch::Tensor axis_convert_mat2_;
  torch::Tensor convert_mat_A2B_;
  torch::Tensor convert_mat_B2A_;

  std::string map_frame_;

  std::deque<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>
    initial_pose_msg_ptr_array_;
  std::mutex initial_pose_array_mtx_;

  LocalizerCore localizer_core_;
};

#endif  // NERF_BASED_LOCALIZER_HPP_
