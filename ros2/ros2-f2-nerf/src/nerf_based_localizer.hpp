#ifndef NERF_BASED_LOCALIZER_HPP_
#define NERF_BASED_LOCALIZER_HPP_

#include <rclcpp/rclcpp.hpp>

class NerfBasedLocalizer : public rclcpp::Node
{
public:
  NerfBasedLocalizer(
    const std::string & name_space = "",
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
};

#endif  // NERF_BASED_LOCALIZER_HPP_
