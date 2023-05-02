#include "nerf_based_localizer.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  std::shared_ptr<NerfBasedLocalizer> node = std::make_shared<NerfBasedLocalizer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
