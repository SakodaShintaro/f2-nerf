#!/usr/bin/env python3

import argparse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
import numpy as np
import cv2
from rclpy.serialization import deserialize_message
import os
import pandas as pd
from util_camera_info import save_camera_info_to_yaml
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
import geometry_msgs
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rosbag2_py
from interpolate import interpolate_pose_in_time
from collections import defaultdict
from typing import Tuple
import yaml


def create_reader(input_bag_dir: str, storage_id: str):
    storage_options = rosbag2_py.StorageOptions(
        uri=input_bag_dir, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader, storage_options, converter_options


def transform_pose_base_link_2_camera(pose: PoseWithCovarianceStamped, target_frame: str, tf_buffer: Buffer) -> Tuple[np.ndarray, np.ndarray]:
    # get static transform
    transform = tf_buffer.lookup_transform(
        target_frame="base_link",
        source_frame=target_frame,
        time=pose.header.stamp)

    # transform pose
    R1: geometry_msgs.msg.Quaternion = transform.transform.rotation
    R1: np.ndarray = Rotation.from_quat([R1.x, R1.y, R1.z, R1.w]).as_matrix()
    t1: geometry_msgs.msg.Vector3 = transform.transform.translation
    t1: np.ndarray = np.array([t1.x, t1.y, t1.z])

    # pose
    R2: geometry_msgs.msg.Quaternion = pose.pose.pose.orientation
    R2: np.ndarray = Rotation.from_quat([R2.x, R2.y, R2.z, R2.w]).as_matrix()
    t2: geometry_msgs.msg.Vector3 = pose.pose.pose.position
    t2: np.ndarray = np.array([t2.x, t2.y, t2.z])

    # transform
    R: np.ndarray = np.dot(R1, R2)
    t: np.ndarray = np.dot(R2, t1) + t2
    q: np.ndarray = Rotation.from_matrix(R).as_quat()
    return t, q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_rosbag", type=str)
    parser.add_argument("calibration_yaml", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--storage_id", type=str,
                        default="mcap", choices=["mcap", "sqlite3"])
    parser.add_argument("--bgr2rgb", action="store_true")
    args = parser.parse_args()

    path_to_rosbag = args.path_to_rosbag
    calibration_yaml = args.calibration_yaml
    skip = args.skip
    output_dir = args.output_dir
    storage_id = args.storage_id
    bgr2rgb = args.bgr2rgb

    with open(calibration_yaml, "r") as input_file:
        calibration_dict = yaml.safe_load(input_file)

    target_frame = ""
    image_topic_names = [
        "/sensing/camera/camera0/image_rect_color/compressed",
        "/sensing/camera/camera1/image_rect_color/compressed",
        "/sensing/camera/camera2/image_rect_color/compressed",
        "/sensing/camera/camera3/image_rect_color/compressed",
        "/sensing/camera/camera4/image_rect_color/compressed",
        "/sensing/camera/camera5/image_rect_color/compressed",
    ]
    camera_info_topic_names = [
        "/sensing/camera/camera0/camera_info",
        "/sensing/camera/camera1/camera_info",
        "/sensing/camera/camera2/camera_info",
        "/sensing/camera/camera3/camera_info",
        "/sensing/camera/camera4/camera_info",
        "/sensing/camera/camera5/camera_info",
    ]
    assert len(image_topic_names) == len(camera_info_topic_names)
    pose_topic_name = "/localization/pose_estimator/pose_with_covariance"
    image_topic_type = CompressedImage()
    pose_topic_type = PoseWithCovarianceStamped()

    reader, storage_options, converter_options = create_reader(
        path_to_rosbag, storage_id)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/rosbag_info.txt", "w") as f:
        f.write(f"{path_to_rosbag}\n")

    bridge = CvBridge()
    tf_buffer = Buffer()

    index_images_all = 0
    prev_image = None
    image_timestamp_lists = defaultdict(list)
    image_lists = defaultdict(list)
    frame_ids = dict()
    pose_timestamp_list = list()
    pose_list = list()
    each_camera_output_dir = f"{output_dir}/each_camera"
    os.makedirs(each_camera_output_dir, exist_ok=True)
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        t /= 1e9
        if topic in image_topic_names:
            image_msg = deserialize_message(data, image_topic_type)
            frame_ids[topic] = image_msg.header.frame_id
            if image_topic_type == Image():
                curr_image = bridge.imgmsg_to_cv2(
                    image_msg, desired_encoding="passthrough")
            elif image_topic_type == CompressedImage():
                curr_image = bridge.compressed_imgmsg_to_cv2(
                    image_msg, desired_encoding="passthrough")
            if bgr2rgb:
                curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            diff = (1 if prev_image is None else np.abs(
                prev_image - curr_image).sum())
            prev_image = curr_image
            index_images_all += 1
            if diff == 0 or index_images_all % skip != 0:
                continue
            image_timestamp_lists[topic].append(t)
            image_lists[topic].append(curr_image)
            if index_images_all >= 500:
                break
        elif topic == pose_topic_name:
            pose_msg = deserialize_message(data, pose_topic_type)
            pose = pose_msg.pose.pose if pose_topic_type == PoseWithCovarianceStamped() else pose_msg.pose
            pose_timestamp_list.append(t)
            pose_list.append(pose)
        elif topic in camera_info_topic_names:
            camera_info = deserialize_message(data, CameraInfo())
            save_name = topic[1:].replace("/", "_")
            save_camera_info_to_yaml(
                camera_info, f"{each_camera_output_dir}/{save_name}.yaml")
            save_camera_info_to_yaml(
                camera_info, f"{output_dir}/camera_info.yaml")
        elif topic == "/tf" or topic == "/tf_static":
            is_static = (topic == "/tf_static")
            tf_msg = deserialize_message(data, TFMessage())
            if is_static:
                print(tf_msg)
            for transform_stamped in tf_msg.transforms:
                if is_static:
                    tf_buffer.set_transform_static(
                        transform_stamped, "default_authority")
                else:
                    tf_buffer.set_transform(
                        transform_stamped, "default_authority")

    columns = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    for topic_name in image_topic_names:
        print(topic_name)
        image_timestamp_list = np.array(image_timestamp_lists[topic_name])
        image_list = np.array(image_lists[topic_name])
        frame_id = frame_ids[topic_name]
        print(frame_id)

        frame_id = frame_id.replace("_optical", "")

        base_link_2_sensor_kit_base_link = calibration_dict["base_link"]["sensor_kit_base_link"]
        sensor_kit_base_link_2_camera = calibration_dict["sensor_kit_base_link"][frame_id]
        print(base_link_2_sensor_kit_base_link)
        print(sensor_kit_base_link_2_camera)
        # {'pitch': -0.02, 'roll': 0.0, 'x': 0.6895, 'y': 0.0, 'yaw': -0.0225, 'z': 2.1}
        # {'pitch': 0.005, 'roll': -0.01, 'x': 0.215, 'y': 0.031, 'yaw': 0.027, 'z': -0.024}
        r_b2s = Rotation.from_euler(
            "xyz", [base_link_2_sensor_kit_base_link["roll"],
                    base_link_2_sensor_kit_base_link["pitch"],
                    base_link_2_sensor_kit_base_link["yaw"]])
        t_b2s = np.array([base_link_2_sensor_kit_base_link["x"],
                          base_link_2_sensor_kit_base_link["y"],
                          base_link_2_sensor_kit_base_link["z"]])
        r_s2c = Rotation.from_euler(
            "xyz", [sensor_kit_base_link_2_camera["roll"],
                    sensor_kit_base_link_2_camera["pitch"],
                    sensor_kit_base_link_2_camera["yaw"]])
        t_s2c = np.array([sensor_kit_base_link_2_camera["x"],
                          sensor_kit_base_link_2_camera["y"],
                          sensor_kit_base_link_2_camera["z"]])
        r_b2c = r_b2s * r_s2c
        t_b2c = t_b2s + r_b2s.apply(t_s2c)
        print(r_b2c.as_euler("xyz", degrees=True))
        print(t_b2c)

        # df_poseを構築
        df_pose = pd.DataFrame(columns=columns)
        for time, pose in zip(pose_timestamp_list, pose_list):
            # pose
            r: geometry_msgs.msg.Quaternion = pose.orientation
            r: Rotation = Rotation.from_quat([r.x, r.y, r.z, r.w])
            t: geometry_msgs.msg.Vector3 = pose.position
            t: np.ndarray = np.array([t.x, t.y, t.z])
            t = t + r.apply(t_b2c)
            r = r_b2c * r
            df_pose.loc[len(df_pose)] = [
                time, *t, *(r.as_quat())
            ]

        # image_timestamp_listから良いものを選ぶ
        min_pose_t = df_pose["timestamp"].min()
        max_pose_t = df_pose["timestamp"].max()
        ok_image_timestamp = (min_pose_t < image_timestamp_list) * \
            (image_timestamp_list < max_pose_t)
        image_timestamp_list = image_timestamp_list[ok_image_timestamp]
        image_list = image_list[ok_image_timestamp]

        # poseを補間
        df_pose = interpolate_pose_in_time(df_pose, image_timestamp_list)

        save_name = topic_name[1:].replace("/", "_")

        os.makedirs(
            f"{each_camera_output_dir}/{save_name}/", exist_ok=True)
        for i, image in enumerate(image_list):
            save_path = f"{each_camera_output_dir}/{save_name}/{i:08d}.png"
            cv2.imwrite(save_path, image)

        df_pose.to_csv(f"{each_camera_output_dir}/{save_name}_pose.tsv",
                       index=True, sep="\t", float_format="%.12f")

        # plot all of trajectory
        save_path = f"{each_camera_output_dir}/{save_name}_plot_pose.png"
        plt.plot(df_pose["x"], df_pose["y"], label=topic_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.legend()

        # draw arrow at last frame
        original_dir = np.array([1, 0, 0])
        final_dir = r.apply(original_dir)
        plt.arrow(df_pose["x"].iloc[-1], df_pose["y"].iloc[-1],
                  final_dir[0], final_dir[1], width=0.1)

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)

        print(f"Saved to {save_path}")

    # 全カメラ分を統合したものを作る
    os.makedirs(f"{output_dir}/images_original/", exist_ok=True)
    index_images_all = 0
    df_pose_all = pd.DataFrame(columns=columns)
    for topic_name in image_topic_names:
        save_name = topic_name[1:].replace("/", "_")
        image_timestamp_list = np.array(image_timestamp_lists[topic_name])
        image_list = np.array(image_lists[topic_name])
        df_pose = pd.read_csv(
            f"{each_camera_output_dir}/{save_name}_pose.tsv", sep="\t")
        df_pose_all = pd.concat([df_pose_all, df_pose], ignore_index=True)
        for i, image in enumerate(image_list):
            save_path = f"{output_dir}/images_original/{index_images_all:08d}.png"
            cv2.imwrite(save_path, image)
            index_images_all += 1
    df_pose_all.to_csv(f"{output_dir}/pose.tsv",
                       index=True, sep="\t", float_format="%.12f")
