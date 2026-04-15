#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/11/19
# @author:aiden
# 手势控制(gesture control)
import os
import cv2
import math
import time
import rclpy
import signal
import threading
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from interfaces.msg import Points
from geometry_msgs.msg import Twist
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from servo_controller.action_group_controller import ActionGroupController
from ros_robot_controller_msgs.msg import BuzzerState, MotorsState, MotorState,PWMServoState, SetPWMServoState

class HandGestureControlNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.image = None
        self.points = []
        self.running = True
        self.left_and_right = 0
        self.up_and_down = 0
        self.last_point = [0, 0]

        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.move = False
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)

        self.create_subscription(Points, '/hand_trajectory/points', self.get_hand_points_callback, 1)

        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        self.mecanum_pub.publish(Twist())
        
        if 'Pro' in self.machine_type:
            self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)  # 舵机控制(servo control)
            self.controller = ActionGroupController(self.create_publisher(ServosPosition, 'servo_controller', 1), '/home/ubuntu/software/arm_pc/ActionGroups')
            self.controller.run_action('camera_up')
            threading.Thread(target=self.hand_gesture_control_arm, daemon=True).start()
        else:
            threading.Thread(target=self.hand_gesture_control, daemon=True).start()

        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def shutdown(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        self.running = False

    def get_hand_points_callback(self, msg):
        points = []
        left_and_right = [0]
        up_and_down = [0]
        if len(msg.points) > 5:
            for i in msg.points:
                if int(i.x) - self.last_point[0] > 0:
                    left_and_right.append(1)
                else:
                    left_and_right.append(-1)
                if int(i.y) - self.last_point[1] > 0:
                    up_and_down.append(1)
                else:
                    up_and_down.append(-1)
                points.extend([(int(i.x), int(i.y))])
                self.last_point = [int(i.x), int(i.y)]
            self.left_and_right = sum(left_and_right)
            self.up_and_down = sum(up_and_down)
            self.points = np.array(points)


    def acker_turn(self,pulse):
        servo_state = PWMServoState()
        servo_state.id = [1]
        servo_state.position = [pulse]
        data = SetPWMServoState()
        data.state = [servo_state]
        data.duration = 0.1
        self.servo_state_pub.publish(data)


    def hand_gesture_control_arm(self):
        while self.running:
            if len(self.points):
                line = cv2.fitLine(self.points, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = int(abs(math.degrees(math.acos(line[0][0]))))
                self.get_logger().info('******%s'%angle)
                if 90 >= angle > 60:
                    if self.up_and_down > 0:
                        self.get_logger().info('👇')
                    else:
                        self.get_logger().info('👆')
                    time.sleep(0.3)
                    self.controller.run_action('hand_control_pick')
                    self.controller.run_action('camera_up')
                elif 30 > angle >= 0:
                    if self.left_and_right > 0:
                        self.get_logger().info('👉')
                    else:
                        self.get_logger().info('👈')
                    time.sleep(0.3)
                    self.controller.run_action('hand_control_place')
                    self.controller.run_action('camera_up')
                self.points = []
            else:
                time.sleep(0.01)

    def hand_gesture_control(self):
        while self.running:
            if len(self.points):
                line = cv2.fitLine(self.points, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = int(abs(math.degrees(math.acos(line[0][0]))))
                self.get_logger().info('******%s'%angle)
                if 90 >= angle > 60:
                    twist = Twist()
                    if self.up_and_down > 0:
                        # self.get_logger().info('👇')
                        twist.linear.x = -0.1
                    else:
                        # self.get_logger().info('👆')
                        twist.linear.x = 0.1
                    if 'Acker' in self.machine_type:
                        self.acker_turn(1500)
                    time.sleep(0.1)
                    self.mecanum_pub.publish(twist)
                    time.sleep(2)

                elif 30 > angle >= 0:
                    twist = Twist()
                    if self.left_and_right > 0:
                        # self.get_logger().info('👉')
                        if 'Acker' in self.machine_type:
                            self.acker_turn(1200)
                            twist.linear.x = 0.2
                        elif 'Mecanum' in self.machine_type:
                            twist.linear.y = -0.1
                        else:
                            twist.angular.z = 0.5
                    else:
                        # self.get_logger().info('👈')
                        if 'Acker' in self.machine_type:
                            self.acker_turn(1850)
                            twist.linear.x = 0.2
                        elif 'Mecanum' in self.machine_type:
                            twist.linear.y = 0.1
                        else:
                            twist.angular.z = 0.5
                    time.sleep(0.1)
                    self.mecanum_pub.publish(twist)
                    time.sleep(2)
                self.move = True
                self.points = []
            else:
                time.sleep(0.01)
                if 'Acker' in self.machine_type and self.move:
                    self.acker_turn(1500)
                    self.move = False
                self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

def main():
    node = HandGestureControlNode('hand_gesture_control')
    rclpy.spin(node)
    node.destroy_node()

if __name__ == "__main__":
    main()
