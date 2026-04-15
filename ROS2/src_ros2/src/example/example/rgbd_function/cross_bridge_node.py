#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/12/18
# @author:aiden
# 过独木桥(cross bridge)
import os
import cv2
import time
import math
import rclpy
import queue
import signal
import threading
import numpy as np
from sdk import common
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position

class CrossBridgeNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        signal.signal(signal.SIGINT, self.shutdown)
        self.running = True
        self.turn = False
        self.plane_high = self.get_parameter('plane_distance').value
        self.debug = self.get_parameter('debug').value
        self.twist = Twist()
        self.image_queue = queue.Queue(maxsize=2)
        self.left_roi = [290, 300, 235, 245]
        self.center_roi = [290, 300, 315, 325]
        self.right_roi = [290, 300, 395, 405]
        self.debug = self.get_parameter('debug').value
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1) # 舵机控制(servo control)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)  # 底盘控制(chassis control)
        self.create_subscription(Image, '/depth_cam/depth0/image_raw', self.depth_callback, 1)

        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()

        self.mecanum_pub.publish(Twist())
        set_servo_position(self.joints_pub, 1, ((10, 300), (5, 500), (4, 170), (3, 40), (2, 665), (1, 500)))
        time.sleep(1)

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def depth_callback(self, ros_depth_image):
        depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                 buffer=ros_depth_image.data)
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像(if the queue is full, discard the oldest image)
            self.image_queue.get()
        # 将图像放入队列(put the image into the queue)
        self.image_queue.put(depth_image)

    def shutdown(self, signum, frame):
        self.running = False
        self.get_logger().info('\033[1;32m%s\033[0m' % "shutdown")

    def get_roi_distance(self, depth_image, roi):
        roi_image = depth_image[roi[0]:roi[1], roi[2]:roi[3]]
        try:
            distance = round(float(np.mean(roi_image[np.logical_and(roi_image > 0, roi_image < 30000)]) / 1000), 3)
        except:
            distance = 0
        return distance

    def move_policy(self, left_distance, center_distance, right_distance):
        if abs(left_distance - self.plane_high) > 0.02:
            if abs(center_distance - self.plane_high) > 0.02:
                self.twist.angular.z = -1.0 #单位：弧度/秒
            else:
                self.twist.angular.z = -0.2
        elif abs(right_distance - self.plane_high) > 0.02:
            if abs(center_distance - self.plane_high) > 0.02:
                self.twist.angular.z = 1.0
            else:
                self.twist.angular.z = 0.2
        else:
            self.twist.angular.z = 0.0
        if abs(left_distance - self.plane_high) > 0.02 and abs(right_distance - self.plane_high) > 0.02 and abs(center_distance - self.plane_high) > 0.02:
            self.twist = Twist()
            #self.running = False
        else:
            self.twist.linear.x = 0.2

        self.mecanum_pub.publish(self.twist)

    def main(self):
        count = 0
        while self.running:
            try:
                depth_image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue
            depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.45), cv2.COLORMAP_JET)

            if self.debug:
                cv2.circle(depth_color_map, (int((self.left_roi[2] + self.left_roi[3]) / 2), int((self.left_roi[0] + self.left_roi[1]) / 2)), 10, (0, 0, 0), -1)
                cv2.circle(depth_color_map, (int((self.center_roi[2] + self.center_roi[3]) / 2), int((self.center_roi[0] + self.center_roi[1]) / 2)), 10, (0, 0, 0), -1)
                cv2.circle(depth_color_map, (int((self.right_roi[2] + self.right_roi[3]) / 2), int((self.right_roi[0] + self.right_roi[1]) / 2)), 10, (0, 0, 0), -1)
                left_distance = self.get_roi_distance(depth_image, self.left_roi)
                center_distance = self.get_roi_distance(depth_image, self.center_roi)
                right_distance = self.get_roi_distance(depth_image, self.right_roi)
                count += 1
                self.get_logger().info(str([left_distance, center_distance, right_distance]))
                if count > 50 and not math.isnan(center_distance):
                    count = 0
                    self.plane_high = center_distance
                    data = {'/**': {'ros__parameters': {'plane_distance': {}}}}
                    data['/**']['ros__parameters']['plane_distance'] = self.plane_high
                    common.save_yaml_data(data, os.path.join(
                        os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')),
                        'config/bridge_plane_distance.yaml'))
                    self.debug = False
            else:
                left_roi = [self.left_roi[0], self.left_roi[1], self.left_roi[2] - 54, self.left_roi[3] - 54] 
                right_roi = [self.right_roi[0], self.right_roi[1], self.right_roi[2] + 54, self.right_roi[3] + 54] 
                cv2.circle(depth_color_map, (int((left_roi[2] + left_roi[3]) / 2), int((left_roi[0] + left_roi[1]) / 2)), 10, (0, 0, 0), -1)
                cv2.circle(depth_color_map, (int((self.center_roi[2] + self.center_roi[3]) / 2), int((self.center_roi[0] + self.center_roi[1]) / 2)), 10, (0, 0, 0), -1)
                cv2.circle(depth_color_map, (int((right_roi[2] + right_roi[3]) / 2), int((right_roi[0] + right_roi[1]) / 2)), 10, (0, 0, 0), -1)
                left_distance = self.get_roi_distance(depth_image, left_roi)
                center_distance = self.get_roi_distance(depth_image, self.center_roi)
                right_distance = self.get_roi_distance(depth_image, right_roi)
                self.get_logger().info(str([left_distance, center_distance, right_distance]))
                if math.isnan(left_distance):
                    left_distance = 0
                if math.isnan(center_distance):
                    center_distance = 0
                if math.isnan(right_distance):
                    right_distance = 0
                self.move_policy(left_distance, center_distance, right_distance)

            cv2.imshow('depth_color_map', depth_color_map)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                self.running = False
        self.mecanum_pub.publish(Twist())
        self.get_logger().info('\033[1;32m%s\033[0m' % 'shutdown')
        rclpy.shutdown()

def main():
    node = CrossBridgeNode('cross_bridge')
    rclpy.spin(node)
    node.destroy_node()

if __name__ == "__main__":
    main()
