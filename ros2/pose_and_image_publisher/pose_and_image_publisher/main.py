import glob
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import argparse
import time
from scipy.spatial.transform import Rotation
import pandas as pd
import tf2_ros
import geometry_msgs.msg


class ImagePosePublisher(Node):

    def __init__(self, data_dir):
        super().__init__('image_pose_publisher')
        self.image_pub = self.create_publisher(Image, 'image', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initial_pose_with_covariance', 10)
        self.bridge = CvBridge()

        self.image_files = sorted(glob.glob(f"{data_dir}/images/*.png"))
        self.from_cams_meta = False
        if self.from_cams_meta:
            self.poses = np.load(f"{data_dir}/cams_meta.npy")
        else:
            self.poses = pd.read_csv(f"{data_dir}/pose.tsv", sep="\t", index_col=0)
            self.image_files = self.image_files[0:len(self.poses)]

        assert len(self.image_files) == len(self.poses), \
            f"Number of images ({len(self.image_files)}) and poses ({len(self.poses)}) do not match."

        self.offset = [0.7, 0.0, 0.1]

        # Publish tf
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_msg = geometry_msgs.msg.TransformStamped()
        self.tf_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_msg.header.frame_id = "base_link"
        self.tf_msg.child_frame_id = "lidar"
        self.tf_msg.transform.translation.x = self.offset[0]
        self.tf_msg.transform.translation.y = self.offset[1]
        self.tf_msg.transform.translation.z = self.offset[2]
        self.tf_msg.transform.rotation.x = 0.0
        self.tf_msg.transform.rotation.y = 0.0
        self.tf_msg.transform.rotation.z = 0.0
        self.tf_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(self.tf_msg)

        self.timer = self.create_timer(0.1, self.timer_callback)  # Publish at 10 Hz
        self.idx = 0

    def timer_callback(self):
        if self.idx >= len(self.image_files) or self.idx >= len(self.poses):
            self.get_logger().info('Finished publishing images and poses.')
            rclpy.shutdown()
            return
        self.get_logger().info(f'Publishing images and poses {self.idx}.')

        # Publish pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        if self.from_cams_meta:
            curr_pose = self.poses[self.idx][0:12].reshape(3, 4)
            pose_msg.pose.pose.position.x = curr_pose[0, 3]
            pose_msg.pose.pose.position.y = curr_pose[1, 3]
            pose_msg.pose.pose.position.z = curr_pose[2, 3]
            rotation_mat = curr_pose[0:3, 0:3]
            r = Rotation.from_matrix(rotation_mat)
            q = r.as_quat()
            pose_msg.pose.pose.orientation.x = q[0]
            pose_msg.pose.pose.orientation.y = q[1]
            pose_msg.pose.pose.orientation.z = q[2]
            pose_msg.pose.pose.orientation.w = q[3]
        else:
            curr_pose = self.poses.iloc[self.idx]
            pose_msg.pose.pose.position.x = curr_pose["x"]
            pose_msg.pose.pose.position.y = curr_pose["y"]
            pose_msg.pose.pose.position.z = curr_pose["z"]
            pose_msg.pose.pose.orientation.x = curr_pose["qx"]
            pose_msg.pose.pose.orientation.y = curr_pose["qy"]
            pose_msg.pose.pose.orientation.z = curr_pose["qz"]
            pose_msg.pose.pose.orientation.w = curr_pose["qw"]
        self.pose_pub.publish(pose_msg)

        # wait for 0.05 sec
        # In the NeRF node, poses are held in a queue and processed when an image is subscribed.
        # If poses and images are published simultaneously, the pose might not be held in the queue and processed before the image is subscribed.
        # Therefore, a delay is introduced between publishing poses and images to ensure the proper processing order.
        time.sleep(0.05)

        # Publish image
        image = cv2.imread(self.image_files[self.idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        msg_image = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.image_pub.publish(msg_image)

        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    args = parser.parse_args()

    node = ImagePosePublisher(args.data_dir)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
