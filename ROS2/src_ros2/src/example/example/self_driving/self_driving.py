#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# 无人驾驶(autonomous driving)
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
from rclpy.node import Node
import sdk.common as common
from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position
from ros_robot_controller_msgs.msg import MotorsState, SetPWMServoState, PWMServoState

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.005, 0.0, 0.0)
        self.param_init()

        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = False
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ['MACHINE_TYPE']
        # signal.signal(signal.SIGINT, self.shutdown)
        self.lane_detect = lane_detect.LaneDetector("yellow")
        self.cmd_vel_topic = '/controller/cmd_vel'

        # Debug Mode
        self.debug_mode = self.get_parameter('debug_mode').value
        if self.debug_mode:
            self.cmd_vel_topic = '/controller/cmd_vel'

        self.mecanum_pub = self.create_publisher(Twist, '%s' % self.cmd_vel_topic, 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1) # 舵机控制(servo control)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # 进入玩法(enter the game)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # 退出玩法(exit the game)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        self.heartbeat = Heart(self, self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response())) # 心跳包(heartbeat package)
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        self.client = self.create_client(Trigger, '/yolo/init_finish')
        self.client.wait_for_service()
        self.start_yolo_client = self.create_client(Trigger, '/yolo/start', callback_group=timer_cb_group)
        self.start_yolo_client.wait_for_service()
        self.stop_yolo_client = self.create_client(Trigger, '/yolo/stop', callback_group=timer_cb_group)
        self.stop_yolo_client.wait_for_service()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolo_client, Trigger.Request())
        time.sleep(1)
        self.display = False
        if self.get_parameter('start').value:
            set_servo_position(self.joints_pub, 1, ((10, 500), (5, 500), (4, 200), (3, 50), (2, 750), (1, 500)))
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        if 'Acker' in self.machine_type:
            servo_state = PWMServoState()
            servo_state.id = [1]
            servo_state.position = [1500]
            data = SetPWMServoState()
            data.state = [servo_state]
            data.duration = 0.02
            self.servo_state_pub.publish(data)

        #self.park_action()
        self.get_logger().info('\033[1;32m [Please wait YOLO START]\033[0m')
        self.client = self.create_client(Trigger, '/yolo/yolo_start_detect')
        self.client.wait_for_service()

        if self.debug_mode:
            threading.Thread(target=self.debug_function, daemon=True).start()
        else:
            threading.Thread(target=self.main, daemon=True).start()
            self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        
    def param_init(self):
        self.start = False
        self.enter = False

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # 停车标识的x像素坐标(obtain the x-pixel coordinate of a parking sign)
        self.park_y = -1  # 停车标识的y像素坐标(obtain the y-pixel coordinate of a parking sign)

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # 开始转弯(start to turn)

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # 右转标志(right turning sign)

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # 停下标识(stopping sign)
        self.start_park = False  # 开始泊车标识(start parking sign)

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # 离斑马线距离(distance to the zebra crossing)
        self.crosswalk_length = 0.1 + 0.3  # 斑马线长度 + 车长(the length of zebra crossing and the robot)

        self.start_slow_down = False  # 减速标识(slowing down sign)
        self.normal_speed = 0.15  # 正常前进速度(normal driving speed)
        self.slow_down_speed = 0.1  # 减速行驶的速度(slowing down speed)

        self.traffic_signs_status = None  # 记录红绿灯状态(record the state of the traffic lights)
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        if hasattr(self, 'heartbeat') and self.heartbeat:
            self.heartbeat.start()
        with self.lock:
            self.start = False
            if self.get_parameter('use_depth_cam').value:
                self.camera = 'depth_cam'
                self.image_sub = self.create_subscription(Image, '/%s/rgb0/image_raw' % self.camera, self.image_callback, 1)  # 摄像头订阅(subscribe to the camera)
            else:
                self.camera = 'usb_cam'
                self.image_sub = self.create_subscription(Image, '/%s/image_raw' % self.camera, self.image_callback, 1)  # 摄像头订阅(subscribe to the camera)
            self.create_subscription(ObjectsInfo, '/yolo/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        if hasattr(self, 'heartbeat') and self.heartbeat:
            self.heartbeat.stop()
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):  # ctrl+c关闭处理(press 'ctrl+c' to close the program)
        self.is_running = False

    def image_callback(self, ros_image):  # 目标检查回调(callback target checking)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像(if the queue is full, remove the oldest image)
            self.image_queue.get()
        # 将图像放入队列(put the image into the queue)
        self.image_queue.put(rgb_image)
    
    # 泊车处理(parking processing)
    def park_action(self):
        if 'Mecanum' in self.machine_type:
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38/0.2)
            self.mecanum_pub.publish(Twist())
        elif 'Acker' in self.machine_type:
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -0.55
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = 0.55
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = -0.55
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            servo_state = PWMServoState()
            servo_state.id = [1]
            servo_state.position = [1500]
            data = SetPWMServoState()
            data.state = [servo_state]
            data.duration = 0.02
            self.servo_state_pub.publish(data)
            twist = Twist()
            twist.linear.x = -0.15
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
        else:

            twist = Twist()
            time.sleep(0.5)
            twist.angular.z = -1.0
            self.mecanum_pub.publish(twist)
            time.sleep(1.1)
            self.mecanum_pub.publish(Twist())

            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.40/0.2)
            self.mecanum_pub.publish(Twist())

            twist = Twist()
            twist.angular.z = 1.0
            self.mecanum_pub.publish(twist)
            time.sleep(1.1)

            self.mecanum_pub.publish(Twist())

    # 摄像头角度调节(Camera angle adjustment)
    def debug_function(self):
        debug_y_thre = 112
        debug_count = 0
        while self.is_running:
            if debug_count >= 50:
                break
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue
            result_image = image.copy()
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.start:
                h, w = image.shape[:2]
                if self.park_y > 0:
                    if  abs(self.park_y - debug_y_thre) < 20: # 在范围内
                        debug_count += 1
                        cv2.putText(result_image,
                                    'Hold the Camera Angle'
                                    ,org=(10, 30),               
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8,
                                    color=(255, 0, 0),          
                                    thickness=2,
                                    lineType=cv2.LINE_AA
                        )
                        cv2.line(result_image,(1,self.park_y),[w-1,self.park_y],(255, 0, 0),2)
                    else:
                        debug_count = 0
                        cv2.putText(result_image,
                                    text=f'Y: {self.park_y}',
                                    org=(10, 30),               
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8,
                                    color=(0, 255, 0),          
                                    thickness=2,
                                    lineType=cv2.LINE_AA
                        )
                        cv2.line(result_image,(1,self.park_y),[w-1,self.park_y],(0, 255, 0),2)
                    self.get_logger().info('center_y: ' + str(self.park_y))
                else:
                    self.get_logger().info('Please put the robot to the stop Place')
            else:
                time.sleep(0.01)
            cv2.imshow('result_image', result_image)
            cv2.waitKey(1)
        self.get_logger().info('You have completed the calibration!!!')
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    def main(self):
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue
            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]
                # self.get_logger().info('\033[1;32m%d,%d\033[0m' %(h, w))  # 输出图像的高和宽(output the height and width, w)
                # 获取车道线的二值化图(obtain the binary image of the lane)
                binary_image = self.lane_detect.get_binary(image)
                # cv2.imshow('binary', binary_image)
                # 检测到斑马线,开启减速标志(if detecting the zebra crossing, start to slow down)
                if 300 < self.crosswalk_distance and not self.start_slow_down:  # 只有离斑马线足够近时才开始减速，通过判断斑马线的y轴像素距离(The robot starts to slow down only when it is close enough to the zebra crossing, which is determined by judging the y-axis pixel distance of the zebra crossing)
                    self.count_crosswalk += 1
                    if self.count_crosswalk == 3:  # 多次判断，防止误检测(judge multiple times to prevent false detection)
                        self.count_crosswalk = 0
                        self.start_slow_down = True  # 减速标识(sign for slowing down)
                        self.count_slow_down = time.time()  # 减速开始时间戳(start timestamp for slowing down)
                else:  # 需要连续检测，否则重置(need to detect continuously, otherwise reset)
                    self.count_crosswalk = 0

                twist = Twist()
                # 减速行驶处理(deceleration processing)
                if self.start_slow_down:
                    if self.traffic_signs_status is not None:
                        # 通过面积判断离灯灯远近，如果太近那么即使是红灯也不会停(Determine the distance to the traffic lights through the area; if the distance is too close, the robot will not stop even if the traffic light is red)
                        area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2])*abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                        # self.get_logger().info('\033[1;32m red light area %d\033[0m' %area)
                        if self.traffic_signs_status.class_name == 'red' and area > 700:  # 如果遇到红灯就停车, 如果太近不停(If the robot detects a red traffic light, it will stop; if the distance is too close, it will not stop)
                            self.mecanum_pub.publish(Twist())
                            self.stop = True
                        elif self.traffic_signs_status.class_name == 'green':  # 遇到绿灯，速度放缓通过(If the traffic light is green, the robot will slow down and pass through)
                            twist.linear.x = self.slow_down_speed
                            self.stop = False
                    if not self.stop:  # 其他非停止的情况速度放缓， 同时计时，时间=斑马线的长度/行驶速度(In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed)
                        twist.linear.x = self.slow_down_speed
                        if time.time() - self.count_slow_down > self.crosswalk_length/twist.linear.x:
                            self.start_slow_down = False
                else:
                    twist.linear.x = self.normal_speed  # 正常速度(normal speed)

                # 检测到 停车标识+斑马线 就减速, 让识别稳定(If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition)
                # self.get_logger().info('crosswalk_distance: ' + str(self.crosswalk_distance))
                if 0 < self.park_x and 160 < self.crosswalk_distance:
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park and 225 < self.crosswalk_distance:  # 离斑马线足够近时就开启停车(When the robot is close enough to the crosswalk, it will start parking)
                        self.mecanum_pub.publish(Twist())
                        self.start_park = True
                        self.stop = True
                        threading.Thread(target=self.park_action).start() #开启停车线程(start the parking thread)
                
                # 右转及停车补线策略(turning right and parking line correction strategy)
                if self.detect_turn_right: #检测到右转标识且离斑马线足够近，就开启右转策略(If the robot detects a right turn sign and is close enough to the crosswalk, it will activate the right turn strategy)
                    if 365 < self.crosswalk_distance:
                        self.detect_turn_right = False
                        self.turn_right = True
                # self.get_logger().info(str(self.turn_right))
                if self.turn_right:
                    #设置检测线的roi(set the ROI of the dectecion line)
                    self.lane_detect.set_roi(((370, 400, 0, 320, 0.7), (310, 340, 0, 320, 0.2), (250, 280, 0, 320, 0.1)))
                    y = self.lane_detect.add_horizontal_line(binary_image)
                    if 0 < y < 350 :
                        roi = [(0, y), (w, y), (w, 0), (0, 0)]
                        cv2.fillPoly(binary_image, [np.array(roi)], [0, 0, 0])  # 将上面填充为黑色，防干扰(fill the above area with black to prevent interference)
                        min_x = cv2.minMaxLoc(binary_image)[-1][0]  #获取最左边的点坐标(get the coordinates of the leftmost point)
                        cv2.line(binary_image, (min_x, y - 60), (w, y - 60), (255, 255, 255), 50)  # 画虚拟线来驱使转弯, 可以通过调整y的大小来控制转弯的时机(Draw a virtual line to drive the turn, the timing of the turn can be controlled by adjusting the value of y)
                        # cv2.imshow('bin_image_turn_right',binary_image)
                    self.lane_detect.set_roi(((320, 350, 0, 320, 0.7), (260, 290, 0, 320, 0.2), (200, 230, 0, 320, 0.1)))
                elif (0 < self.park_x or self.have_turn_right) and not self.start_turn:  # 检测到停车标识需要填补线，使其保持直走(If the robot detects a stop sign and needs to fill in the line to keep going straight, it will do so)
                    if not self.detect_far_lane:
                        up, down, center = self.lane_detect.add_vertical_line_near(binary_image)
                        self.get_logger().info('detect far lane' + str([center]))
                        binary_image[:, :] = 0  # 全置黑，防止干扰(set all to black to prevent interference)
                        if 100 < center < 150:  # 当将要看不到车道线时切换到识别较远车道线(When the lane is about to be out of sight, switch to detecting the far lane)
                            self.detect_far_lane = True
                    else:
                        up, down = self.lane_detect.add_vertical_line_far(binary_image)
                        binary_image[:, :] = 0
                    # self.get_logger().info('up,down' + str(up) + str(down))
                    if up != down:
                        cv2.line(binary_image, up, down, (255, 255, 255), 20)  # 手动画车道线(manually draw the lane)

                result_image, lane_angle, lane_x, max_area = self.lane_detect(binary_image, image.copy())  # 在处理后的图上提取车道线中心(Obtain the center of the lane on the processed image)
                # self.get_logger().info('lane_x' + str(lane_x))
                # cv2.imshow('image', image)
                # 巡线处理(line following processing)
                if not self.stop:
                    if self.turn_right:
                        # self.get_logger().info('lane_x,max_area' + str([lane_x, max_area]))
                        if max_area > 1000:
                            if self.count_turn == 1:
                                self.lane_detect.set_roi(((370, 400, 0, 320, 0.7), (320, 350, 0, 320, 0.2), (270, 290, 0, 320, 0.1)))
                            self.count_turn += 1
                            if self.count_turn > 1 and not self.start_turn:  # 稳定转弯(stable turning)
                                self.start_turn = True
                                self.turn_right = False
                                self.count_turn = 0
                                self.start_turn_time_stamp = time.time()
                            twist.angular.z = -0.60  # 转弯速度(turning speed)
                        else:  # 直道由pid计算转弯修正(use PID algorithm to correct turns on a straight road)
                            self.count_turn = 0
                            if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                                if 'Acker' in self.machine_type:
                                    if time.time() - self.start_turn_time_stamp > 5:
                                        self.turn_right = False
                                        self.start_turn = False
                                else:
                                    self.turn_right = False
                                    self.start_turn = False
                            if not self.start_turn:
                                self.pid.SetPoint = 90  # 在车道中间时线的坐标(the coordinate of the line while the robot is in the middle of the lane)
                                if abs(lane_x - 90) < 20:
                                    lane_x = 90
                                self.pid.update(lane_x)
                                twist.angular.z = common.set_range(self.pid.output, -0.8, 0.8)
                    else:
                        # self.get_logger().info('lane_x' + str(lane_x))
                        if lane_x > 115:
                            self.count_turn += 1
                            if self.count_turn > 5 and not self.start_turn:  # 稳定转弯(stable turning)
                                self.start_turn = True
                                self.count_turn = 0
                                self.start_turn_time_stamp = time.time()
                            twist.angular.z = -0.50  # 转弯速度(turning speed)
                        else:  # 直道由pid计算转弯修正(use PID algorithm to correct turns on a straight road)
                            self.count_turn = 0
                            if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                                self.start_turn = False
                            if not self.start_turn:
                                self.pid.SetPoint = 90  # 在车道中间时线的坐标(the coordinate of the line while the robot is in the middle of the lane)
                                if abs(lane_x - 90) < 20:
                                    lane_x = 90
                                self.pid.update(lane_x)
                                twist.angular.z = common.set_range(self.pid.output, -0.8, 0.8)
                    self.mecanum_pub.publish(twist)
                    if not self.start:
                        self.mecanum_pub.publish(Twist())
                else:
                    self.pid.clear()

                # 绘制识别的物体，由于物体检测的速度小于线检测的速度，所以绘制的框会有所偏离(When drawing the recognized object, the drawn box may deviate from the actual position due to the slower speed of object detection compared to line detection)
                if self.objects_info != []:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )
            else:
                time.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                cv2.imshow('result', bgr_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # 按q或者esc退出(press Q or Esc to quit)
                    self.is_running = False
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    # 获取目标检测结果(Obtain the target detection result)
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:  # 没有识别到时重置变量(If it is not recognized, reset the variable)
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
            self.park_y = -1
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                if class_name == 'crosswalk':  
                    if center[1] > min_distance:  # 获取最近的人行道y轴像素坐标(Obtain recent y-axis pixel coordinate of the crosswalk)
                        min_distance = center[1]
                elif class_name == 'right':  # 获取右转标识(obtain the right turning sign)
                    area = abs(i.box[0] - i.box[2])*abs(i.box[1] - i.box[3])
                    if area > 200:
                        if not self.turn_right:
                            self.count_right += 1
                            self.count_right_miss = 0
                            if self.count_right >= 1:  # 检测到多次就将右转标志至真(If it is detected multiple times, take the right turning sign to true)
                                self.have_turn_right = True
                                self.detect_turn_right = True
                                self.count_right = 0
                    else:
                        self.get_logger().info('right tag too far')
                elif class_name == 'park':  # 获取停车标识中心坐标(obtain the center coordinate of the parking sign)
                    area = abs(i.box[0] - i.box[2])*abs(i.box[1] - i.box[3])
                    if area > 150:
                        self.park_x = center[0]
                    self.park_y = center[1]
                elif class_name == 'red' or class_name == 'green':  # 获取红绿灯状态(obtain the status of the traffic light)
                    self.traffic_signs_status = i
        
            self.crosswalk_distance = min_distance

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()


