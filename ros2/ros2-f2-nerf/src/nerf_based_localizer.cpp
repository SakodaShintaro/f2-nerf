#include "nerf_based_localizer.hpp"

#include <rclcpp/rclcpp.hpp>

NerfBasedLocalizer::NerfBasedLocalizer(
  const std::string & name_space, const rclcpp::NodeOptions & options)
: Node("nerf_based_localizer", name_space, options)
{
  RCLCPP_INFO(this->get_logger(), "nerf_based_localizer is created.");
}
