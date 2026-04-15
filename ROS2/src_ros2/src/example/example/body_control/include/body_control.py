#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/11/07
# @author:aiden
# 肌体控制(body control)
import os
import cv2
import time
import rclpy
import queue
import signal
import threading
import numpy as np
import faulthandler
import sdk.fps as fps
import mediapipe as mp
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import BuzzerState, MotorsState, MotorState,PWMServoState, SetPWMServoState
from servo_controller_msgs.msg import ServosPosition
from servo_controller.action_group_controller import ActionGroupController
faulthandler.enable()

mp_pose = mp.solutions.pose

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP

RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP

LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE

RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE

def get_joint_landmarks(img, landmarks):
    """
    将landmarks从medipipe的归一化输出转为像素坐标(Convert landmarks from medipipe's normalized output to pixel coordinates)
    :param img: 像素坐标对应的图片(picture corresponding to pixel coordinate)
    :param landmarks: 归一化的关键点(normalized keypoint)
    :return:
    """
    h, w, _ = img.shape
    landmarks = [(lm.x * w, lm.y * h) for lm in landmarks]
    return np.array(landmarks)

def joint_distance(landmarks):
    distance_list = []

    d1 = landmarks[LEFT_HIP] - landmarks[LEFT_SHOULDER]
    d2 = landmarks[LEFT_HIP] - landmarks[LEFT_WRIST]
    dis1 = d1[0]**2 + d1[1]**2
    dis2 = d2[0]**2 + d2[1]**2
    distance_list.append(round(dis1/dis2, 1))
   
    d1 = landmarks[RIGHT_HIP] - landmarks[RIGHT_SHOULDER]
    d2 = landmarks[RIGHT_HIP] - landmarks[RIGHT_WRIST]
    dis1 = d1[0]**2 + d1[1]**2
    dis2 = d2[0]**2 + d2[1]**2
    distance_list.append(round(dis1/dis2, 1))
    
    d1 = landmarks[LEFT_HIP] - landmarks[LEFT_ANKLE]
    d2 = landmarks[LEFT_ANKLE] - landmarks[LEFT_KNEE]
    dis1 = d1[0]**2 + d1[1]**2
    dis2 = d2[0]**2 + d2[1]**2
    distance_list.append(round(dis1/dis2, 1))
   
    d1 = landmarks[RIGHT_HIP] - landmarks[RIGHT_ANKLE]
    d2 = landmarks[RIGHT_ANKLE] - landmarks[RIGHT_KNEE]
    dis1 = d1[0]**2 + d1[1]**2
    dis2 = d2[0]**2 + d2[1]**2
    distance_list.append(round(dis1/dis2, 1))
    
    return distance_list

class BodyControlNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.drawing = mp.solutions.drawing_utils
        self.body_detector = mp_pose.Pose(
            static_image_mode=False,
            min_tracking_confidence=0.7,
            min_detection_confidence=0.7)
        self.running = True
        self.fps = fps.FPS()  # fps计算器(FPS calculator)
        signal.signal(signal.SIGINT, self.shutdown)

        self.move_finish = True
        self.stop_flag = False
        self.left_hand_count = []
        self.right_hand_count = []
        self.left_leg_count = []
        self.right_leg_count = []

        self.detect_status = [0, 0, 0, 0]
        self.move_status = [0, 0, 0, 0]
        self.last_status = 0
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)

        self.machine_type = os.environ.get('MACHINE_TYPE')

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)  # 底盘控制(chassis control)

        camera = 'depth_cam'
        self.create_subscription(Image, '/depth_cam/rgb0/image_raw' , self.image_callback, 1)
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 1)
        self.motor_pub = self.create_publisher(MotorsState, '/ros_robot_controller/set_motor', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)

        if 'Pro' in  self.machine_type:
            self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
            self.controller = ActionGroupController(self.create_publisher(ServosPosition, 'servo_controller', 1), '/home/ubuntu/software/arm_pc/ActionGroups')
            self.client = self.create_client(Trigger, '/controller_manager/init_finish')
            self.client.wait_for_service()
            self.controller.run_action('camera_up')
        self.mecanum_pub.publish(Twist())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def shutdown(self, signum, frame):
        self.running = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像(if the queue is full, discard the oldest image)
            self.image_queue.get()
        # 将图像放入队列(put the image into the queue)
        self.image_queue.put(rgb_image)


    def acker_turn(self,pulse):
        servo_state = PWMServoState()
        servo_state.id = [1]
        servo_state.position = [pulse]
        data = SetPWMServoState()
        data.state = [servo_state]
        data.duration = 0.1
        self.servo_state_pub.publish(data)

    def move(self, *args):
        if args[0].angular.z == -1.0:
            self.acker_turn(1900)
            time.sleep(0.2)
            motor1 = MotorState()
            motor1.id = 2
            motor1.rps = 2.0
            motor2 = MotorState()
            motor2.id = 4
            motor2.rps = -0.5
            msg = MotorsState()
            msg.data = [motor1, motor2]
            self.motor_pub.publish(msg)
            time.sleep(8.0)
            self.acker_turn(1500)
            motor1 = MotorState()
            motor1.id = 2
            motor1.rps = 0.0
            motor2 = MotorState()
            motor2.id = 4
            motor2.rps = 0.0
            msg = MotorsState()
            msg.data = [motor1, motor2]
            self.motor_pub.publish(msg)
        elif args[0].angular.z == 1.0:
            self.acker_turn(1100)
            time.sleep(0.2)
            motor1 = MotorState()
            motor1.id = 2
            motor1.rps = 0.5
            motor2 = MotorState()
            motor2.id = 4
            motor2.rps = -2.0
            msg = MotorsState()
            msg.data = [motor1, motor2]
            self.motor_pub.publish(msg)
            time.sleep(8.0)
            self.acker_turn(1500)
            motor1 = MotorState()
            motor1.id = 2
            motor1.rps = 0.0
            motor2 = MotorState()
            motor2.id = 4
            motor2.rps = 0.0
            msg = MotorsState()
            msg.data = [motor1, motor2]
            self.motor_pub.publish(msg)
        else:
            self.mecanum_pub.publish(args[0])
            time.sleep(args[1])
            self.mecanum_pub.publish(Twist())
            time.sleep(0.1)
        self.stop_flag =True
        self.move_finish = True

    def buzzer_warn(self):
        msg = BuzzerState()
        msg.freq = 1900
        msg.on_time = 0.2
        msg.off_time = 0.01
        msg.repeat = 1
        self.buzzer_pub.publish(msg)

    def image_proc(self, image):
        image_flip = cv2.flip(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1)
        results = self.body_detector.process(image)
        if results is not None and results.pose_landmarks is not None:
            if self.move_finish:
                twist = Twist()
                landmarks = get_joint_landmarks(image, results.pose_landmarks.landmark)
                distance_list = (joint_distance(landmarks))
              
                if distance_list[0] < 1:
                    self.detect_status[0] = 1
                if distance_list[1] < 1:
                    self.detect_status[1] = 1
                if 0 < distance_list[2] < 2:
                    self.detect_status[2] = 1
                if 0 < distance_list[3] < 2:
                    self.detect_status[3] = 1
                
                self.left_hand_count.append(self.detect_status[0])
                self.right_hand_count.append(self.detect_status[1])
                self.left_leg_count.append(self.detect_status[2])
                self.right_leg_count.append(self.detect_status[3])                   
                #print(distance_list) 
               
                if len(self.left_hand_count) == 4:
                    count = [sum(self.left_hand_count), 
                             sum(self.right_hand_count), 
                             sum(self.left_leg_count), 
                             sum(self.right_leg_count)]

                    self.left_hand_count = []
                    self.right_hand_count = []
                    self.left_leg_count = []
                    self.right_leg_count = []
                
                    if self.stop_flag:
                        if count[self.last_status - 1] <= 1:
                            self.stop_flag = False
                            self.move_status = [0, 0, 0, 0]
                            self.buzzer_warn()
                    else:
                        if count[0] > 2:
                            self.move_status[0] = 1
                        if count[1] > 2:
                            self.move_status[1] = 1
                        if count[2] > 2:
                            self.move_status[2] = 1
                        if count[3] > 2:
                            self.move_status[3] = 1
                        
                        if self.move_status[0]:
                            self.move_finish = False
                            self.last_status = 1
                            if 'Mecanum' in self.machine_type:
                                twist.linear.y = -0.3
                            elif 'Acker' in self.machine_type:
                                twist.angular.z = -1.0
                            elif 'Differential' in self.machine_type:
                                twist.linear.x = -0.3
                            threading.Thread(target=self.move, args=(twist, 1)).start()
                        elif self.move_status[1]:
                            self.move_finish = False
                            self.last_status = 2
                            if 'Mecanum' in self.machine_type:
                                twist.linear.y = 0.3
                            elif 'Acker' in self.machine_type:
                                twist.angular.z = 1.0
                            elif 'Differential' in self.machine_type:
                                twist.linear.x = 0.3
                            threading.Thread(target=self.move, args=(twist, 1)).start()

                        elif self.move_status[2]:
                            self.move_finish = False
                            self.last_status = 3
                            twist.linear.x = 0.3
                            threading.Thread(target=self.move, args=(twist, 1)).start()
                        elif self.move_status[3]:
                            self.move_finish = False
                            self.last_status = 4
                            twist.linear.x = -0.3
                            threading.Thread(target=self.move, args=(twist, 1)).start()

                self.detect_status = [0, 0, 0, 0]
            
            result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.drawing.draw_landmarks(
                result_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            return cv2.flip(result_image, 1)
        else:
            return image_flip

    def main(self):
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue
            try:
                result_image = self.image_proc(np.copy(image))
            except BaseException as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % e)
                result_image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1)
            self.fps.update()
            result_image = self.fps.show_fps(result_image)
            cv2.imshow(self.name, result_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 按q或者esc退出(press Q or Esc to exit)
                self.mecanum_pub.publish(Twist())
                self.running = False

        rclpy.shutdown()

def main():
    node = BodyControlNode('body_control')
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
