import rclpy
from rclpy.node import Node

from tier4_localization_msgs.srv import PoseWithCovarianceStamped
from std_srvs.srv import SetBool


class PoseReflectorService(Node):
    def __init__(self):
        super().__init__('pose_reflector_service_node')

        # Create the service
        self.srv = self.create_service(
            PoseWithCovarianceStamped, '/localization/pose_estimator/ndt_align_srv', self.reflect_pose_callback)

        self.srv_trigger = self.create_service(
            SetBool, "/localization/pose_estimator/trigger_node", self.srv_trigger_callback
        )

    def reflect_pose_callback(self, request, response):
        # Just reflect the pose back in the response
        response.success = True
        response.pose_with_covariance = request.pose_with_covariance
        return response

    def srv_trigger_callback(self, request, response):
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)

    pose_reflector_service = PoseReflectorService()

    rclpy.spin(pose_reflector_service)

    pose_reflector_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
