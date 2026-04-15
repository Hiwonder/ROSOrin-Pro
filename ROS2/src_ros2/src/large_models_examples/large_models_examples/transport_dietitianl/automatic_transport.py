#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/11/18
# @author:aiden
# 追踪拾取(tracking and picking) - AprilTag 版本 (取消放置对齐版，增加快递拾取/放置)
import os
import ast
import cv2
import time
import math
import queue
import rclpy
import threading
import numpy as np
from sdk import common
from sdk.pid import PID
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, Empty
from app.common import ColorPicker
from sensor_msgs.msg import Image
from interfaces.msg import Pose2D
from geometry_msgs.msg import Twist, Point
from xf_mic_asr_offline import voice_play
from servo_controller_msgs.msg import ServosPosition
from interfaces.srv import SetPose2D, SetPoint, SetBox
from servo_controller.bus_servo_control import set_servo_position
from servo_controller.action_group_controller import ActionGroupController

# 引入 AprilTag 检测器
from dt_apriltags import Detector

class AutomaticTransportNode(Node):
    # 原本的配置路径
    config_path = '/home/ubuntu/ros2_ws/src/large_models_examples/config/automatic_transport_roi.yaml'
    # 新增的快递配置路径
    # config_exp_path = '/home/ubuntu/ros2_ws/src/large_models_examples/config/automatic_transport_exp.yaml'

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        
        # 初始化 AprilTag 检测器
        self.at_detector = Detector(searchpath=['apriltags'], families='tag36h11', nthreads=4, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
        
        # 将原本的颜色阈值替换为目标 Tag ID, -1 表示没有目标
        self.target_tag_id = -1
        
        self.box = []
        
        self.mouse_click = False
        self.selection = None  # 实时跟踪鼠标的跟踪区域
        self.track_window = None  # 要检测的物体所在区域
        self.drag_start = None  # 标记，是否开始拖动鼠标
        self.start_circle = True
        self.start_click = False
        self.pick_finish = False
        self.place_finish = False

        self.running = True
        self.detect_count = 0
        
        # 状态标志位
        self.start_pick = False
        self.start_place = False
        self.start_pick_exp = False
        
        self.pick = False
        self.place = False
        self.pick_exp = False
        
        self.linear_base_speed = 0.007
        self.angular_base_speed = 0.03

        self.yaw_pid = PID(P=0.025, I=0, D=0.000)
        self.linear_pid = PID(P=0.001, I=0, D=0)
        self.angular_pid = PID(P=0.001, I=0, D=0)

        self.linear_speed = 0
        self.angular_speed = 0
        self.yaw_angle = 90

        self.pick_stop_x = 320  #320
        self.pick_stop_y = 388
        self.place_stop_x = 320
        self.place_stop_y = 388
        self.stop = True

        self.d_y = 15
        self.d_x = 15


        self.status = "approach"
        self.count_stop = 0
        self.count_turn = 0

        self.declare_parameter('status', 'start')
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.display_box = True
        self.start_time = time.time()
        self.start = self.get_parameter('start').value
        self.enable_display = self.get_parameter('enable_display').value
        self.debug = self.get_parameter('debug').value
        self.image_name = 'image'

        self.language = os.environ['ASR_LANGUAGE']
        self.machine_type = os.environ['MACHINE_TYPE']

        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.image_pub = self.create_publisher(Image, '~/image_result', 1)

        self.create_subscription(Image, 'depth_cam/rgb0/image_raw', self.image_callback, 1)
        
        # 原有的服务
        self.create_service(Trigger, '~/pick', self.start_pick_callback)
        self.create_service(Trigger, '~/place', self.start_place_callback) 
        # 新增的快递服务
        self.create_service(Trigger, '~/pick_exp', self.start_pick_exp_callback)
        self.create_service(Trigger, '~/place_exp', self.start_place_exp_callback) 
        
        self.create_service(SetPoint, '~/set_target_color', self.set_target_color_srv_callback)
        self.create_service(SetBox, '~/set_box', self.set_box_srv_callback)

        self.controller = ActionGroupController(self.create_publisher(ServosPosition, 'servo_controller', 1), '/home/ubuntu/software/arm_pc/ActionGroups')
        self.get_logger().info("Action Group Controller has been started")
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        
        self.action_finish_pub = self.create_publisher(Bool, '~/action_finish', 1)
        
        self.mecanum_pub.publish(Twist())
        set_servo_position(self.joints_pub, 2, ((1, 500), (2, 701), (3, 120), (4, 88), (5, 500), (10, 500)))
        time.sleep(2)
        
        self.get_logger().info("Automatic Pick Node has been started")
        
        if self.debug == 'pick':
            self.controller.run_action('pick_basket_debug')
            time.sleep(5)
            set_servo_position(self.joints_pub, 1, ((1, 500), (2, 534), (3, 107), (4, 334), (5, 125), (10, 200)))
            time.sleep(0.5)
            set_servo_position(self.joints_pub, 1, ((1, 500), (2, 701), (3, 120), (4, 88), (5, 500), (10, 200)))
            time.sleep(1)
            msg = Trigger.Request()
            self.start_pick_callback(msg, Trigger.Response())
        elif self.debug == 'place':
            self.controller.run_action('place_basket_debug')
            time.sleep(5)
            set_servo_position(self.joints_pub, 1, ((1, 500), (2, 500), (3, 122), (4, 506), (5, 125), (10, 500)))
            time.sleep(0.5)
            set_servo_position(self.joints_pub, 1, ((1, 500), (2, 701), (3, 120), (4, 88), (5, 500), (10, 200)))
            time.sleep(1)
            msg = Trigger.Request()
            self.start_place_callback(msg, Trigger.Response())
        elif self.debug == 'pick_exp':
            # 新增 pick_exp 的调试
            self.controller.run_action('pick_exp_debug') # 若有专用的调试动作组可以修改这里
            time.sleep(5)
            set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 246), (3, 425), (4, 152), (5, 143), (10, 200)))
            time.sleep(0.5)
            set_servo_position(self.joints_pub, 1, ((1, 500), (2, 736), (3, 15), (4, 180), (5, 500), (10, 200)))
            time.sleep(1)
            msg = Trigger.Request()
            self.start_pick_exp_callback(msg, Trigger.Response())

        threading.Thread(target=self.action_thread, daemon=True).start()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        return response

    def set_box_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_box')
        self.box = [request.x_min, request.y_min, request.x_max, request.y_max]
        self.display_box = True
        self.start_time = time.time()
        self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_box"
        return response

    def set_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_target_color cleared (not using color)')
        # 这里仅作清空处理
        self.target_tag_id = -1
        self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "clear_target"
        return response

    # ================= 原始拾取 =================
    def start_pick_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start pick (basket)")
        self.place_finish = False
        self.linear_speed = 0
        self.angular_speed = 0
        self.yaw_angle = 90

        # 从原来的 roi 配置文件读取
        param = self.get_parameter('pick_stop_pixel_coordinate').value
        self.get_logger().info('\033[1;32mget pick stop pixel coordinate: %s\033[0m' % str(param))
        self.pick_stop_x = param[0]
        self.pick_stop_y = param[1]
        self.stop = True
        self.d_y = 15
        self.d_x = 15

        self.pick = False
        self.place = False
        self.pick_exp = False
        self.start_pick_exp = False

        self.status = "approach"
        self.count_stop = 0
        self.count_turn = 0
        self.linear_pid.clear()
        self.angular_pid.clear()
        
        self.start_pick = True
        self.target_tag_id = 1  
        response.success = True
        response.message = "start_pick"
        return response 

    # ================= 新增：快递拾取 =================
    def start_pick_exp_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start pick (exp)")
        self.place_finish = False
        self.linear_speed = 0
        self.angular_speed = 0
        self.yaw_angle = 90

        set_servo_position(self.joints_pub, 1.0, ((1, 500), (2, 736), (3, 15), (4, 180), (5, 500), (10, 200)))
        time.sleep(1.0)

        # 尝试从 exp 配置文件读取参数，如果参数服务器没有预设，可以给个默认值
        try:
            param = self.get_parameter('pick_exp_stop_pixel_coordinate').value
            self.pick_stop_x = param[0]
            self.pick_stop_y = param[1]
        except:
            self.pick_stop_x = 320
            self.pick_stop_y = 388
            
        self.get_logger().info('\033[1;32mget pick_exp stop pixel coordinate: [%s, %s]\033[0m' % (self.pick_stop_x, self.pick_stop_y))
        
        self.stop = True
        self.d_y = 15
        self.d_x = 15

        self.pick = False
        self.place = False
        self.pick_exp = False
        self.start_pick = False

        self.status = "approach"
        self.count_stop = 0
        self.count_turn = 0
        self.linear_pid.clear()
        self.angular_pid.clear()
        
        self.start_pick_exp = True
        # 【重要】在这里设置要去拾取快递的目标 Tag ID，例如设置为 3
        self.target_tag_id = 0  
        response.success = True
        response.message = "start_pick_exp"
        return response 

    # ================= 原始放置 =================
    def start_place_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start place (直接执行，无视觉对齐)")
        self.pick_finish = False
        self.linear_speed = 0
        self.angular_speed = 0
        self.d_y = 10
        self.d_x = 10
        self.stop = True
        self.pick = False
        self.pick_exp = False
        
        self.start_place = False 
        self.place = True        

        self.linear_pid.clear()
        self.angular_pid.clear()
        self.target_tag_id = -1
        response.success = True
        response.message = "start_place"
        return response 

    # ================= 新增：快递放置 =================
    def start_place_exp_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start place_exp (直接执行，无视觉对齐)")
        self.pick_finish = False
        self.linear_speed = 0
        self.angular_speed = 0
        self.d_y = 10
        self.d_x = 10
        self.stop = True
        self.pick = False
        self.pick_exp = False
        self.place = False
        
        self.start_place = False 
        
        # 放置不需要视觉追踪，置为 -1
        self.target_tag_id = -1

        # 直接执行预设动作（和原来放置一致，可以直接复用 self.place 或新加逻辑）
        # 这里为了区分，可以直接调用控制器
        threading.Thread(target=self.place_exp_action, daemon=True).start()

        response.success = True
        response.message = "start_place_exp"
        return response 

    def place_exp_action(self):
        """专门给快递放置的动作"""
        self.mecanum_pub.publish(Twist())
        time.sleep(1)
        # 如果需要别的动作组可以替换 'place_basket' 为对应的快递动作组文件
        self.controller.run_action('place_exp') 
        time.sleep(8)
        self.get_logger().info('place_exp finish')
        msg = Bool()
        msg.data = True
        self.action_finish_pub.publish(msg)
        self.place_finish = True

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()


    def apriltag_detect(self, img):
        """AprilTag 识别"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)
        
        center_x, center_y, angle = -1, -1, -1
        
        for tag in tags:
            # 只处理我们设定的目标 Tag ID
            if tag.tag_id == self.target_tag_id:
                corners = tag.corners.astype(int)
                cv2.drawContours(img, [corners], -1, (0, 255, 255), 2, cv2.LINE_AA)
                
                # 获取最小外接矩形
                rect = cv2.minAreaRect(np.array(tag.corners).astype(np.float32))
                (center, (width, height), angle) = rect
                center_x, center_y = int(center[0]), int(center[1])
                
                # 绘制中心点
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)
                break
                
        return center_x, center_y, angle

    def action_thread(self):
        while True:
            # ==== 原始 pick ====
            if self.pick:
                self.target_tag_id = -1 
                self.start_pick = False
                self.mecanum_pub.publish(Twist())
                time.sleep(0.5)

                # set_servo_position(self.joints_pub, 1, ((1, 500), (2, 532), (3, 127), (4, 316), (5, 500), (10, 500)))
                # time.sleep(1)
                # set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 534), (3, 107), (4, 334), (5, 125), (10, 200)))
                # time.sleep(0.5)
                # set_servo_position(self.joints_pub, 0.8, ((1, 500), (2, 356), (3, 67), (4, 654), (5, 125), (10, 200)))
                # time.sleep(0.8)
                # set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 356), (3, 67), (4, 654), (5, 125), (10, 650)))
                # time.sleep(1.0)
                # set_servo_position(self.joints_pub, 2.0, ((1, 500), (2, 765), (3, 20), (4, 375), (5, 125), (10, 650)))
                # time.sleep(2.0)

                self.controller.run_action('pick_basket')
                time.sleep(6)
                
                self.pick = False
                self.get_logger().info('pick finish')
                msg = Bool()
                msg.data = True
                self.action_finish_pub.publish(msg)
                self.pick_finish = True
                
            # ==== 快递 pick_exp ====
            elif self.pick_exp:
                self.target_tag_id = -1 
                self.start_pick_exp = False
                self.mecanum_pub.publish(Twist())

                # 这里执行的动作如果需要不一样，可以修改参数。暂时和前面一样
                # time.sleep(0.5)
                # set_servo_position(self.joints_pub, 1, ((1, 500), (2, 532), (3, 127), (4, 316), (5, 500), (10, 500)))
                # time.sleep(1)
                # set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 534), (3, 107), (4, 334), (5, 125), (10, 200)))
                # time.sleep(0.5)
                # set_servo_position(self.joints_pub, 0.8, ((1, 500), (2, 356), (3, 67), (4, 654), (5, 125), (10, 200)))
                # time.sleep(0.8)
                # set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 356), (3, 67), (4, 654), (5, 125), (10, 650)))
                # time.sleep(1.0)
                # set_servo_position(self.joints_pub, 2.0, ((1, 500), (2, 765), (3, 20), (4, 375), (5, 125), (10, 650)))
                # time.sleep(2.0)
                self.controller.run_action('pick_exp')
                time.sleep(6)
                
                self.pick_exp = False
                self.get_logger().info('pick_exp finish')
                msg = Bool()
                msg.data = True
                self.action_finish_pub.publish(msg)
                self.pick_finish = True

            # ==== 原始 place ====
            elif self.place:
                self.target_tag_id = -1
                self.start_place = False
                self.mecanum_pub.publish(Twist())
                time.sleep(1)
                self.controller.run_action('place_basket')
                time.sleep(8)
                self.get_logger().info('place finish')
                self.place = False
                msg = Bool()
                msg.data = True
                self.action_finish_pub.publish(msg)
                self.place_finish = True
            else:
                time.sleep(0.01)

    def pick_handle(self, image, is_exp=False):
            """合并处理原始拾取与快递拾取，is_exp 标志是否写入 exp 的 yaml 文件"""
            twist = Twist()

            check_pick = self.pick_exp if is_exp else self.pick
            debug_str = 'pick_exp' if is_exp else 'pick'
            cfg_path =  self.config_path

            if not check_pick or self.debug == debug_str:
                object_center_x, object_center_y, object_angle = self.apriltag_detect(image) 
                if self.debug == debug_str:
                    self.detect_count += 1
                    if self.detect_count > 10:
                        self.detect_count = 0
                        self.pick_stop_y = object_center_y
                        self.pick_stop_x = object_center_x
                        
                        data = common.get_yaml_data(cfg_path)
                        # 如果配置文件为空则创建
                        if data is None or '/**' not in data:
                            data = {'/**': {'ros__parameters': {}}}
                            
                        key_name = 'pick_exp_stop_pixel_coordinate' if is_exp else 'pick_stop_pixel_coordinate'
                        data['/**']['ros__parameters'][key_name] = [self.pick_stop_x, self.pick_stop_y]
                        common.save_yaml_data(data, cfg_path)
                        self.debug = False
                    self.get_logger().info(f'[{debug_str}] x_y: ' + str([object_center_x, object_center_y])) 
                elif object_center_x > 0:
                    ########电机pid处理#########
                    # 1. 计算前后移动 (X轴)
                    self.linear_pid.SetPoint = self.pick_stop_y
                    if abs(object_center_y - self.pick_stop_y) <= self.d_y:
                        object_center_y = self.pick_stop_y
                    if self.status != "align":
                        self.linear_pid.update(object_center_y)  
                        output = self.linear_pid.output
                        tmp = math.copysign(self.linear_base_speed, output) + output
                        self.linear_speed = tmp
                        if tmp > 0.4:
                            self.linear_speed = 0.4
                        if tmp < -0.4:
                            self.linear_speed = -0.4
                        if abs(tmp) <= 0.0075:
                            self.linear_speed = 0

                    # 2. 计算左右偏离量 (现在用于Y轴平移)
                    self.angular_pid.SetPoint = self.pick_stop_x
                    if abs(object_center_x - self.pick_stop_x) <= self.d_x:
                        object_center_x = self.pick_stop_x
                    if self.status != "align":
                        self.angular_pid.update(object_center_x)  
                        output = self.angular_pid.output
                        tmp = math.copysign(self.angular_base_speed, output) + output
                        
                        self.angular_speed = tmp
                        if tmp > 1.5:
                            self.angular_speed = 1.5
                        if tmp < -1.5:
                            self.angular_speed = -1.5
                        if abs(tmp) <= 0.038:
                            self.angular_speed = 0
                            
                    # 3. 计算角度偏差 (用于Z轴自转对齐)
                    if self.status != "align":
                        # 避免 45 度时的跳变
                        if object_angle < 40: 
                            object_angle += 90
                        
                        self.yaw_pid.SetPoint = 90
                        if abs(object_angle - 90) <= 3:
                            object_angle = 90
                        self.yaw_pid.update(object_angle)  
                        # 限制一下旋转的速度，免得在追踪时转得太猛
                        yaw_output = self.yaw_pid.output
                        if yaw_output > 0.5:
                            yaw_output = 0.5
                        elif yaw_output < -0.5:
                            yaw_output = -0.5
                        self.yaw_angle = yaw_output


                    if abs(self.linear_speed) == 0 and abs(self.angular_speed) == 0:
                        if self.machine_type == 'JetRover_Mecanum':
                            self.count_turn += 1
                            if self.count_turn >= 3:
                                self.count_turn = 3
                                self.status = "align"
                                if self.count_stop < 3: 
                                    if object_angle < 40: 
                                        object_angle += 90
                                    self.yaw_pid.SetPoint = 90
                                    if abs(object_angle - 90) <= 3:
                                        object_angle = 90
                                    self.yaw_pid.update(object_angle)  
                                    self.yaw_angle = self.yaw_pid.output
                                    if object_angle != 90:
                                        if abs(self.yaw_angle) <=0.038:
                                            self.count_stop += 1
                                        twist.linear.y = float(-2 * 0.3 * math.sin(self.yaw_angle / 2))
                                        twist.angular.z = float(self.yaw_angle)
                                    else:
                                        self.count_stop += 1
                                elif self.count_stop <=6:
                                    self.d_x = 5
                                    self.d_y = 5
                                    self.count_stop += 1
                                    self.status = "adjust"
                                else:
                                    self.count_stop = 0
                                    if is_exp:
                                        self.pick_exp = True
                                    else:
                                        self.pick = True
                        else:
                            self.count_stop += 1
                            if self.count_stop > 15:
                                self.count_stop = 0
                                if is_exp:
                                    self.pick_exp = True
                                else:
                                    self.pick = True
                    else:
                        if self.count_stop >= 3:
                            self.count_stop = 3
                        self.count_turn = 0
                        
                        if self.status != 'align':
                            # X轴：前后靠近
                            twist.linear.x = float(self.linear_speed) 
                            
                            # Y轴：左右平移保持目标居中
                            lateral_speed = float(self.angular_speed) # 注意这里的正负号可能需要实测调试
                            if lateral_speed > 0.15:
                                lateral_speed = 0.15
                            elif lateral_speed < -0.15:
                                lateral_speed = -0.15
                            twist.linear.y = lateral_speed
                            
                            # Z轴：在追踪的同时，逐渐调整车体角度去对齐 Tag
                            if object_angle != 90:
                                twist.angular.z = float(self.yaw_angle)
                            else:
                                twist.angular.z = 0.0

            self.mecanum_pub.publish(twist)

            return image

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(cv2.resize(rgb_image, (640, 480)))

    def main(self):
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue
            
            result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 判断是否有 target_tag_id (放置阶段设为 -1，自动跳过视觉追踪)
            if self.target_tag_id != -1:
                if self.start_pick:
                    self.stop = True
                    result_image = self.pick_handle(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), is_exp=False)
                elif self.start_pick_exp:
                    self.stop = True
                    result_image = self.pick_handle(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), is_exp=True)
                else:
                    if self.stop:
                        self.stop = False
                        self.mecanum_pub.publish(Twist())
                    result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
                cv2.line(result_image, (self.pick_stop_x, self.pick_stop_y - 10), (self.pick_stop_x, self.pick_stop_y + 10), (0, 255, 255), 2)
                cv2.line(result_image, (self.pick_stop_x - 10, self.pick_stop_y), (self.pick_stop_x + 10, self.pick_stop_y), (0, 255, 255), 2)
            
            if self.enable_display:
                cv2.imshow(self.image_name, result_image)
                cv2.waitKey(1)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "bgr8"))

        set_servo_position(self.joints_pub, 2, ((1, 500), (2, 701), (3, 120), (4, 88), (5, 500), (10, 200)))
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

def main():
    node = AutomaticTransportNode('automatic_transport')
    rclpy.spin(node)
    node.destroy_node()
 
if __name__ == "__main__":
    main()