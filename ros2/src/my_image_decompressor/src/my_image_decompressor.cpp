#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>

class ImageDecompressor : public rclcpp::Node
{
public:
  ImageDecompressor() : Node("my_image_decompressor")
  {
    rclcpp::QoS qos(10);
    qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/src_topic", qos,
      std::bind(&ImageDecompressor::listener_callback, this, std::placeholders::_1));
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/dst_topic", 10);
  }

private:
  void listener_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    cv::Mat cv_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg();
    RCLCPP_INFO_STREAM(this->get_logger(), "Subscribe: " << msg->header.frame_id.c_str());
    publisher_->publish(*image_msg);
  }
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageDecompressor>());
  rclcpp::shutdown();
  return 0;
}
