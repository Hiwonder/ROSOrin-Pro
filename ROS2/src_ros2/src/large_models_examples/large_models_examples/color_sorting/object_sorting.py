#!/usr/bin/env python3
# coding: utf8
# Object Sorting 目标分拣
import os
import cv2
import yaml
import time
import math
import copy
import queue
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sdk import fps
from app.common import Heart
from dt_apriltags import Detector
from interfaces.srv import SetStringBool
from kinematics_msgs.srv import SetRobotPose, SetJointValue
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from kinematics.kinematics_control import set_pose_target, set_joint_value_target
from large_models_examples.color_sorting.utils import calculate_grasp_yaw, position_change_detect, pick_and_place, image_process, distortion_inverse_map, common

class ObjectSortingNode(Node):
    place_position = {'green': [-0.006, 0.20, 0.020],
                      'red': [0.064, 0.20, 0.020],
                      'blue': [-0.076, 0.20, 0.020],
                      'tag1': [-0.076, 0.15, 0.020],
                      'tag2': [-0.006, 0.15, 0.020],
                      'tag3': [0.064, 0.15, 0.020]}
    
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        proto_path = '/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/color_sorting/hed_model/deploy.prototxt'
        model_path = '/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/color_sorting/hed_model/hed_pretrained_bsds.caffemodel'
        self.image_process = image_process.GetObjectSurface(proto_path, model_path)
        self.at_detector = Detector(searchpath=['apriltags'],
               families='tag36h11',
               nthreads=4,
               quad_decimate=1.0,
               quad_sigma=0.0,
               refine_edges=1,
               decode_sharpening=0.25,
               debug=0)
        self.lock = threading.RLock()
        self.fps = fps.FPS()    # frame rate counter 帧率统计器
        self.bridge = CvBridge()  # Used for the conversion between ROS Image messages and OpenCV images. 用于ROS Image消息与OpenCV图像之间的转换
        self.image_queue = queue.Queue(maxsize=2)
        self.config_file = 'transform.yaml'
        self.calibration_file = 'calibration_color.yaml'
        self.config_path = "/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/color_sorting/config/"
        self.data = common.get_yaml_data(os.path.join(self.config_path, "lab_config.yaml"))  
        self.lab_data = self.data['/**']['ros__parameters']
        # self.camera_type = os.environ['DEPTH_CAMERA_TYPE']
        self.camera_type = 'USB_CAM'
        
        self.tag_size = 0.025
        self.min_area = 500
        self.max_area = 7000
        self.target_labels = {
            "red": False,
            "green": False,
            "blue": False,
            "tag1": False,
            "tag2": False,
            "tag3": False,
        }
        self.running = True
        # Initialize basic parameters. 初始化基本参数
        self._init_parameters()

        # sub
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result',  1)

        self.timer_cb_group = ReentrantCallbackGroup()
        # services and topics
        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_sorting_srv = self.create_service(SetBool, '~/enable_sorting', self.enable_sorting_srv_callback)
        self.set_target_srv = self.create_service(SetStringBool, '~/set_target', self.set_target_srv_callback)
        self.kinematics_client = self.create_client(SetRobotPose, 'kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()
        self.set_joint_value_target_client = self.create_client(SetJointValue, 'kinematics/set_joint_value_target', callback_group=self.timer_cb_group)
        self.set_joint_value_target_client.wait_for_service()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=self.timer_cb_group)

    def get_node_state(self, request, response):
        response.success = True
        return response

    def _init_parameters(self):
        self.heart = None
        self.endpoint = None
        self.target_miss_count = 0
        self.transport_info = None
        self.intrinsic = None
        self.distortion = None
        self.start_transport = False
        self.enable_sorting = False
        self.white_area_center = None
        self.enter = False
        self.roi = []
        self.count_move = 0
        self.count_still = 0
        self.target = None
        self.start_get_roi = False
        self.last_position = None
        self.last_object_info_list = None
        self.image_sub = None
        self.camera_info_sub = None
        self.data = common.get_yaml_data(os.path.join(self.config_path, "lab_config.yaml"))  
        self.lab_data = self.data['/**']['ros__parameters']

    def init_process(self):
        self.timer.cancel()
        
        threading.Thread(target=self.main, daemon=True).start()
        threading.Thread(target=self.transport_thread, daemon=True).start()
        if self.get_parameter('start').value:
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            req = SetBool.Request()
            req.data = True 
            res = SetBool.Response()
            self.enable_sorting_srv_callback(req, res)

        if not self.get_parameter('broadcast').value:
            target_list = ["red", "green", "blue"]
            req = SetStringBool.Request()
            req.data_bool = True
            for i in target_list:
                req.data_str = i
                res = SetBool.Response()
                self.set_target_srv_callback(req, res)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'init finish')

    def go_home(self, interrupt=True):
        if self.target is not None:
            if self.target[0] in ["bule", "tag1"]:
                t = 1.6
        elif self.target is not None:
            if self.target[0] in ["green", "tag2"]:
                t = 1.3
        elif self.target is not None:
            if self.target[0] in ["red", "tag3"]:
                t = 1.0
        else :
            t = 1.0
        if interrupt:
            set_servo_position(self.joints_pub, 0.5, ((10, 200), ))
            time.sleep(0.5)
        joint_angle = [500, 520, 185, 50, 500]
        # joint_angle = [879, 523, 185, 50, 500]
        # joint_angle = [500, 420, 225, 116, 500]
        # joint_angle = [500, 470, 220, 90, 500]  #jetrover

        set_servo_position(self.joints_pub, 1, ((2, joint_angle[1]), (3, joint_angle[2]), (4, joint_angle[3]), (5, 500)))
        time.sleep(1)

        set_servo_position(self.joints_pub, 1, ((1, joint_angle[0]), ))
        time.sleep(1)

    def get_roi(self):
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # Convert to numpy array. 转换为 numpy 数组
            extristric = np.array(config['extristric'])
            corners = np.array(config['corners']).reshape(-1, 3)
            self.white_area_center = np.array(config['white_area_pose_world'])
        while True:
            intrinsic = self.intrinsic
            distortion = self.distortion
            if intrinsic is not None and distortion is not None:
                break
            time.sleep(0.1)

        tvec = extristric[:1]  # Take first row. 取第一行
        rmat = extristric[1:]  # Take next three rows. 取后面三行

        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03)
        self.extristric = tvec, rmat
        imgpts, jac = cv2.projectPoints(corners[:-1], np.array(rmat), np.array(tvec), intrinsic, distortion)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Crop RIO region. 裁切出ROI区域
        x_min = min(imgpts, key=lambda p: p[0])[0] # The minimum value of X-axis. x轴最小值
        x_max = max(imgpts, key=lambda p: p[0])[0] # The maximum value of X-axis. x轴最大值
        y_min = min(imgpts, key=lambda p: p[1])[1] # The minimum value of Y-axis. y轴最小值
        y_max = max(imgpts, key=lambda p: p[1])[1] # The maximum value of Y-axis. y轴最大值
        roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
            
        self.roi = roi

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "enter object sorting")
        self._init_parameters()
        self.heart = Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)
        for k, v in self.target_labels.items():
            self.target_labels[k] = False
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb0/image_raw', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb0/camera_info', self.camera_info_callback, 1)
        self.start_get_roi = True
        joint_angle = [500, 520, 185, 50, 500] 

        # joint_angle = [879, 520, 185, 50, 500]  #综合颜色和垃圾分类参数
        # joint_angle = [500, 420, 225, 116, 500]
        # joint_angle = [500, 470, 220, 90, 500]  #jetrover
        set_servo_position(self.joints_pub, 1, ((1, 500), (2, joint_angle[1]), (3, joint_angle[2]), (4, joint_angle[3]), (5, 500), (10, 200)))

        self.enter = True
        
        response.success = True
        response.message = "start"
        return response
     
    def exit_srv_callback(self, request, response):
        if self.enter:
            self.get_logger().info('\033[1;32m%s\033[0m' % "exit  object sorting")
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if self.camera_info_sub is not None:
                self.destroy_subscription(self.camera_info_sub)
                self.camera_info_sub = None
            self.heart.destroy()
            self.heart = None
            self.enter = False
            self.start_transport = False
            pick_and_place.interrupt(True)
        
        response.success = True
        response.message = "start"
        return response
    
    def enable_sorting_srv_callback(self, request, response):
        if request.data:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'start  object sorting')
            pick_and_place.interrupt(False)
            self.enable_sorting = True
        else:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'stop  object sorting')
            pick_and_place.interrupt(True)
            self.enable_sorting = False
        
        response.success = True
        response.message = "start"
        return response

    def set_target_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mset target %s %s\033[0m' % (str(request.data_str), str(request.data_bool)))
        if request.data_str in self.target_labels:
            self.target_labels[request.data_str] = request.data_bool

        response.success = True
        response.message = "start"
        return response

    def get_object_pixel_position(self, bgr_image, roi):
        target_info = []
        draw_image = bgr_image.copy()
        roi_img = bgr_image[roi[0]:roi[1], roi[2]:roi[3]]

        if roi_img.size == 0:
            self.get_logger().warn(f"ROI 裁切为空! ROI 坐标: {roi}")
            return draw_image, []  # 提前返回，避免崩溃

        # roi_img = self.image_process.get_top_surface(roi_img)
        # cv2.imshow('roi_img', roi_img)
        image_lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)  # Convert to LAB space. 转换到 LAB 空间

        for i in ['red', 'green', 'blue']:
            index = 0
            mask = cv2.inRange(image_lab, tuple(self.lab_data['color_range_list'][i]['min']), tuple(self.lab_data['color_range_list'][i]['max']))  # Binaryzation 二值化
            # Smooth edges, remove small blocks, and merge adjacent blocks. 平滑边缘，去除小块，合并靠近的块
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Find all contours. 找出所有轮廓
            contours_area = map(lambda c: (math.fabs(cv2.contourArea(c)), c), contours)  # Calculate contour area. 计算轮廓面积
            contours = map(lambda a_c: a_c[1], filter(lambda a: self.min_area <= a[0] <= self.max_area, contours_area))
            for c in contours:
                rect = cv2.minAreaRect(c)  # Obtain the minimum bounding rectangle. 获取最小外接矩形
                (center_x, center_y), _ = cv2.minEnclosingCircle(c)
                center_x, center_y = roi[2] + center_x, roi[0] + center_y
                cv2.circle(draw_image, (int(center_x), int(center_y)), 8, (0, 0, 0), -1)
                # self.get_logger().info(f'识别到物体: {i}, 像素中心点: ({int(center_x)}, {int(center_y)})')
                corners = list(map(lambda p: (roi[2] + p[0], roi[0] + p[1]), cv2.boxPoints(rect)))  # Obtain the four corner points of the minimum rectangle and convert to the coordinates of the original image. 获取最小外接矩形的四个角点, 转换回原始图的坐标
                cv2.drawContours(draw_image, [np.intp(corners)], -1, (0, 255, 255), 2, cv2.LINE_AA)  # Draw rectangle contour. 绘制矩形轮廓
                
                cv2.line(draw_image, (int(center_x), 0), (int(center_x), 480), (255, 255, 0), 2, cv2.LINE_AA)   #几乎一直都有青色的线
                index += 1  # Incremental numbering 序号递增
                angle = int(round(rect[2]))  # Rectangle angle 矩形角度
                # self.get_logger().info(f'>>>  {i}, 2D角度: {angle}')
                target_info.append([i, index, (int(center_x), int(center_y)), (int(rect[1][0]), int(rect[1][1])), angle])
        return draw_image, target_info

    def get_object_world_position(self, position, intrinsic, extristric, white_area_center, height=0.035):
        projection_matrix = np.row_stack((np.column_stack((extristric[1], extristric[0])), np.array([[0, 0, 0, 1]])))
        world_pose = common.pixels_to_world([position], intrinsic, projection_matrix)[0]
        world_pose[0] = -world_pose[0]
        world_pose[1] = -world_pose[1]
        position = white_area_center[:3, 3] + world_pose
        position[2] = height
        
        config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
        offset = tuple(config_data['pixel']['offset'])
        scale = tuple(config_data['pixel']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i]
            position[i] = position[i] + offset[i]

        # # 保存旧的 X 和 Y
        # final_x_old = position[0]
        # final_y_old = position[1]

        # # 机械臂向右转90度 = 物体跑到机械臂右边
        # # New X (前方) = Old Y (左方)
        # position[0] = -final_y_old
        # # New Y (左方) = -Old X (后方，即负的前方)
        # position[1] = final_x_old

        return position, projection_matrix

    def calculate_pick_grasp_yaw(self, position, target, target_info, intrinsic, projection_matrix):
        yaw = math.degrees(math.atan2(position[1], position[0]))
        if position[0] < 0 and position[1] < 0:
            yaw = yaw + 180
        elif position[0] < 0 and position[1] > 0:
            yaw = yaw - 180
        # 0.09x0.015
        # self.get_logger().info(f'>>> {position[1]:.4f}), 位置Yaw: {yaw:.4f} 度')
        # self.get_logger().info(f'>>> Yaw: {yaw:.4f} 度')

        gripper_size = [common.calculate_pixel_length(0.09, intrinsic, projection_matrix),
                        common.calculate_pixel_length(0.015, intrinsic, projection_matrix)]

        return calculate_grasp_yaw.calculate_gripper_yaw_angle(target, target_info, gripper_size, yaw)

    def calculate_place_grasp_yaw(self, position, angle=0):
        yaw = math.degrees(math.atan2(position[1], position[0]))
        if position[0] < 0 and position[1] < 0:
            yaw = yaw + 180
        elif position[0] < 0 and position[1] > 0:
            yaw = yaw - 180
        yaw1 = yaw + angle
        if yaw < 0:
            yaw2 = yaw1 + 90
        else:
            yaw2 = yaw1 - 90

        yaw = yaw2
        if abs(yaw1) < abs(yaw2):
            yaw = yaw1
        yaw = 500 + int(yaw / 240 * 1000)

        return yaw

    def transport_thread(self):
        while self.running:
            if self.start_transport:
                position, yaw, target = self.transport_info
                if position[0] > 0.22:
                    position[2] += 0.01
                config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                offset = tuple(config_data['kinematics']['offset'])
                scale = tuple(config_data['kinematics']['scale'])
                for i in range(3):
                    position[i] = position[i] * scale[i]
                    position[i] = position[i] + offset[i]
                # self.get_logger().info(f'pick2:{position}')
                finish = pick_and_place.pick(position, 80, yaw, 540, 0.02, self.joints_pub, self.kinematics_client)
                if finish:
                    position = copy.deepcopy(self.place_position[target[0]])

                    yaw = self.calculate_place_grasp_yaw(position, 0)
                    config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                    offset = tuple(config_data['kinematics']['offset'])
                    scale = tuple(config_data['kinematics']['scale'])
                    angle = math.degrees(math.atan2(position[1], position[0]))
                    if angle > 45:
                        # self.get_logger().info(f'1:{position}')
                        position = [position[0] * scale[1], position[1] * scale[0], position[2] * scale[2]]
                        position = [position[0] - offset[1], position[1] + offset[0], position[2] + offset[2]]
                    elif angle < -45:
                        # self.get_logger().info(f'2:{position}')
                        position = [position[0] * scale[1], position[1] * scale[0], position[2] * scale[2]]
                        position = [position[0] + offset[1], position[1] - offset[0], position[2] + offset[2]]
                    else:
                        # self.get_logger().info(f'3:{position}')
                        position = [position[0] * scale[0], position[1] * scale[1], position[2] * scale[2]]
                        position = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]

                    # self.get_logger().info(f'{position}')
                    finish = pick_and_place.place(position, 80, yaw, 200, self.joints_pub, self.kinematics_client)
                    # finish = pick_and_place.place(position, 45, yaw, 200, self.joints_pub, self.kinematics_client)
                    if finish:
                        self.go_home(False)
                    else:
                        self.go_home(True)
                else:
                    self.go_home(True)
                self.target = None
                self.start_transport = False
            else:
                time.sleep(0.1)

    def main(self):
        while self.running:
            if self.enter:
                try:
                    bgr_image = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                
                if self.start_get_roi:
                    self.get_roi()
                    self.start_get_roi = False
                roi = self.roi.copy()
                intrinsic = self.intrinsic
                if len(roi) > 0 and self.enable_sorting and not self.start_transport:
                    display_image, target_info = self.get_object_pixel_position(bgr_image, roi)
                    
                    tags = self.at_detector.detect(cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY), True, (intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]), self.tag_size)
                    if len(tags) > 0:
                        index = 0
                        for tag in tags:
                            if 'tag%d'%tag.tag_id in self.target_labels:
                                corners = tag.corners.astype(int)
                                cv2.drawContours(display_image, [corners], -1, (0, 255, 255), 2, cv2.LINE_AA)
                                rect = cv2.minAreaRect(np.array(tag.corners).astype(np.float32))
                                # rect includes (center point, (width, height), rotation Angle). rect 包含 (中心点, (宽度, 高度), 旋转角度)
                                (center, (width, height), angle) = rect
                                index += 1
                                target_info.append(['tag%d'%tag.tag_id, index, (int(center[0]), int(center[1])), (int(width), int(height)), angle])
                    if target_info:
                        if self.last_object_info_list:
                            # Reorder by comparing the positions of the objects from the last time. 对比上一次的物体的位置来重新排序
                            target_info = position_change_detect.position_reorder(target_info, self.last_object_info_list, 20)
                    self.last_object_info_list = copy.deepcopy(target_info)
                    for target in target_info:
                        cv2.putText(display_image, '{}'.format(target[0]),(target[2][0] - 4 * len(target[0] + str(target[1])), target[2][1] + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    target_miss = True 
                    for target in target_info:  # detect

                        if self.target_labels[target[0]]:  # app set
                            if self.target is not None:  # If there is already a target, skip the other objects directly. 如果已经有了目标，其他物体就直接跳过
                                if self.target[0] != target[0] or self.target[1] != target[1]:
                                    continue
                                else:
                                    target_miss = False
                                    self.target = target

                            # if self.camera_type == 'USB_CAM':
                            #     x, y = distortion_inverse_map.undistorted_to_distorted_pixel(target[2][0], target[2][1], self.intrinsic, self.distortion)
                            #     target[2] = (x, y)

                            position, projection_matrix = self.get_object_world_position(target[2], intrinsic, self.extristric, self.white_area_center)
                            result = self.calculate_pick_grasp_yaw(position, target, target_info, intrinsic, projection_matrix)
                            if result is not None and self.target is None:
                                # self.get_logger().info(f"!!! 首次锁定目标: {target[0]} ({target[1]}) !!!")
                                self.target = target
                                break

                            if self.last_position is not None and self.target is not None and result is not None:
                                e_distance = round(math.sqrt(pow(self.last_position[0] - position[0], 2)) + math.sqrt(
                                    pow(self.last_position[1] - position[1], 2)), 5)
                                # self.get_logger().info(f'e_distance: {e_distance}')
                                if e_distance <= 0.005:  # The Euclidean distance is less than 2mm to prevent the object from being picked up while it is still moving. 欧式距离小于2mm, 防止物体还在移动时就去夹取了
                                    cv2.line(display_image, result[1][0], result[1][1], (255, 255, 0), 2, cv2.LINE_AA)
                                    self.count_move = 0
                                    self.count_still += 1
                                else:
                                    self.count_move += 1
                                    self.count_still = 0

                                if self.count_move > 10:
                                    self.target = None
                                if self.count_still > 10:
                                    self.count_still = 0
                                    self.count_move = 0
                                    self.target = target
                                    # self.get_logger().info(f'>>>{target[0]}, 最终抓取角度 (度): {result[0]}')
                                    yaw = 500 + int(result[0] / 240 * 1000)
                                    # self.get_logger().info(f'>>>{target[0]}, 最终舵机值 (Yaw): {yaw}')
                                    self.transport_info = [position, yaw, target]
                                    self.start_transport = True   #暂时注释掉
                            self.last_position = position
                    if target_miss:
                        self.target_miss_count += 1
                    if self.target_miss_count > 10:
                        self.target_miss_count = 0
                        self.target = None
                else:
                    display_image = bgr_image.copy()
                # self.fps.update()
                # display_image = self.fps.show_fps(display_image)
                if bgr_image is not None and self.get_parameter('display').value:
                    cv2.imshow('result_image', display_image)
                    cv2.waitKey(1)
                self.result_publisher.publish(self.bridge.cv2_to_imgmsg(display_image, "bgr8"))
            else:
                time.sleep(0.1)

    def camera_info_callback(self, msg):
        # 从 CameraInfo.p 中取出 3x4 矩阵，再取左上角 3x3 作为去畸变后图像的内参
        P = np.array(msg.p, dtype=np.float32).reshape(3, 4)
        new_k = P[:, :3]  # 3x3
        # 直接用 3x3 矩阵,保持和原来 np.matrix(3x3) 一样的 2D 形状
        self.intrinsic = np.matrix(new_k)
        # 图像已经经过 undistort_node 去畸变，所以这里的畸变系数设为 0
        self.distortion = np.zeros(5, dtype=np.float32)

    def image_callback(self, ros_rgb_image):
        # Convert the image from ros format to opencv format. 将ros格式图像转换为opencv格式
        cv_image = self.bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # Discard the oldest image if the queue is full. 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # Put image into the queue. 将图像放入队列
        self.image_queue.put((bgr_image))

def main():
    node = ObjectSortingNode('object_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.running = False  # Stop thread flag. 停止线程标志
        executor.shutdown()

if __name__ == "__main__":
    main()
