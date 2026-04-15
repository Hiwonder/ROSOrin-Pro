#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import cv2
import math
import yaml
import copy
import time
import torch
import queue
import rclpy
import threading
import numpy as np
import sdk.fps as fps
import message_filters
# from sdk import common
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String, Float32, Bool
from std_srvs.srv import Trigger, SetBool, Empty
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.utils.ops import scale_masks
from tf2_ros import Buffer, TransformListener, TransformException

# from speech import speech
from app.common import Heart
from large_models.config import *
from large_models_msgs.srv import SetString, SetModel, SetBox
from kinematics_msgs.srv import SetRobotPose, SetJointValue
from kinematics.kinematics_control import set_pose_target, set_joint_value_target
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from large_models_examples.color_sorting.utils import utils, image_process, calculate_grasp_yaw, pick_and_place, common

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def prompt(results, bboxes=None, points=None, labels=None, texts=None, log=None):
    if bboxes is None and points is None and texts is None:
        return results
    prompt_results = []
    if not isinstance(results, list):
        results = [results]
    for result in results:
        if len(result) == 0:
            prompt_results.append(result)
            continue
        #强制 Mask 掩码和原始图片一样
        masks = result.masks.data
        if masks.shape[1:] != result.orig_shape:
            masks = scale_masks(masks[None], result.orig_shape)[0]
        idx = torch.zeros(len(result), dtype=torch.bool, device=device)
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
            full_mask_areas = torch.sum(masks, dim=(1, 2))
 
            u = mask_areas / full_mask_areas  
            u = torch.nan_to_num(u, nan=0.0) 
            indices = (u >= (torch.max(u) - 0.1)).nonzero(as_tuple=True)[1] 
            u1 = full_mask_areas / bbox_areas
            max_index = indices[torch.argmax(u1[indices])]
            idx[max_index] = True

        prompt_results.append(result[idx])

    return prompt_results

class ObjectTransport(Node):
    hand2cam_tf_matrix = [
        [0.0, 0.0, 1.0, -0.101],  # x
        [-1.0, 0.0, 0.0, 0.01],  # y
        [0.0, -1.0, 0.0, 0.05],  # z
        [0.0, 0.0, 0.0, 1.0]
    ]
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.fps = fps.FPS() # 帧率统计器(frame rate counter)
        self.image_queue = queue.Queue(maxsize=2)
        self._init_parameters()

        self.depth_cam_type = os.environ['DEPTH_CAMERA_TYPE']

        self.set_above = False
        self.record_position = []
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        self.gripper_pixel_size = utils.get_gripper_size(500)
        self.base_gripper_height = utils.get_gripper_size(500)[1]

        self.get_logger().info("Initializing publishers...")
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.transport_finished_pub = self.create_publisher(Bool, '~/transport_finished', 1)
        
        self.get_logger().info("Initializing callback groups and clients...")
        timer_cb_group = ReentrantCallbackGroup()
        self.set_joint_value_target_client = self.create_client(SetJointValue, 'kinematics/set_joint_value_target', callback_group=timer_cb_group)
        self.set_joint_value_target_client.wait_for_service()

        
        self.kinematics_client = self.create_client(SetRobotPose, 'kinematics/set_pose_target')       
        self.kinematics_client.wait_for_service()

        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_sorting_srv = self.create_service(SetBool, '~/enable_transport', self.enable_transport_srv_callback)
        self.set_pick_position_srv = self.create_service(SetBox, '~/set_pick_position', self.set_pick_position_srv_callback)
        self.set_place_position_srv = self.create_service(SetBox, '~/set_place_position', self.set_place_position_srv_callback)
        self.record_position_srv = self.create_service(SetBox, '~/record_position', self.record_position_srv_callback)

        self.get_logger().info("Setting up FastSAM model...")
        code_path = os.path.abspath(os.path.split(os.path.realpath(__file__))[0])
        overrides = dict(conf=0.4, task="segment", mode="predict", model=os.path.join(os.path.dirname(code_path), 'resources/models', "FastSAM-x.pt"), save=False, imgsz=(640))

        self.predictor = FastSAMPredictor(overrides=overrides)

        self.get_logger().info("Warming up FastSAM model...")
        self.predictor(np.zeros((640, 480, 3), dtype=np.uint8))
        self.get_logger().info("...FastSAM model warmed up.")

        self.get_logger().info("Loading configuration files...")
        self.config_file = 'transform.yaml'
        self.calibration_file = 'calibration_transport.yaml'
        self.config_path = "/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/color_sorting/config/"
        
        self.get_logger().info("Reading YAML data...")
        self.data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/color_sorting/config/lab_config.yaml")  
        self.get_logger().info("...YAML data loaded.")

        self.lab_data = self.data['/**']['ros__parameters']
        
        self.get_logger().info("Initializing TF listener...")
        tf_buffer = Buffer()
        self.tf_listener = TransformListener(tf_buffer, self)
        
        self.get_logger().info("Setting up async TF wait (this line should not block).")
        # tf_future = tf_buffer.wait_for_transform_async(
        #     target_frame='usb_link',
        #     source_frame='depth_cam_color_frame',
        #     time=rclpy.time.Time()
        # )

        # rclpy.spin_until_future_complete(self, tf_future)
        # try:
        #     transform = tf_buffer.lookup_transform(
        #         'depth_cam_color_frame', 'usb_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5.0) )
        #     self.static_transform = transform  # 保存变换数据
        #     self.get_logger().info(f'Static transform: {self.static_transform}')
        # except TransformException as e:
        #     self.get_logger().error(f'Failed to get static transform: {e}')

        # # 提取平移和旋转
        # translation = transform.transform.translation
        # rotation = transform.transform.rotation

        # self.transform_matrix = common.xyz_quat_to_mat([translation.x, translation.y, translation.z], [rotation.w, rotation.x, rotation.y, rotation.z])
        # self.hand2cam_tf_matrix = np.matmul(self.transform_matrix, self.hand2cam_tf_matrix)


        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response


    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键
            self.mouse_click = True
            self.drag_start = (x, y)  # 记录起始位置
            self.drag_type = 'pick'   # 记录拖动类型
            self.track_window = None
            self.selection = None

        elif event == cv2.EVENT_RBUTTONDOWN: # 按下右键 (用于 Place)
            self.mouse_click = True
            self.drag_start = (x, y)  # 记录起始位置
            self.drag_type = 'place'  # 记录拖动类型
            self.track_window = None
            self.selection = None

        elif event == cv2.EVENT_MBUTTONDOWN: # 按下中键 (用于清空)
            self.mouse_click = False
            self.selection = None
            self.track_window = None
            self.drag_start = None
            self.drag_type = None
        
        if self.drag_start:  # 如果正在拖动
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax) # 实时更新选框
        
        if event == cv2.EVENT_LBUTTONUP:  # 松开左键
            if self.drag_start and self.selection and self.drag_type == 'pick':
                self.track_window = self.selection # 最终选框
                with self.lock:
                    # self.get_logger().info(f'鼠标框选, 添加到抓取队列: {self.track_window}')
                    self.action_list.append(['pick', self.track_window])
            
            self.mouse_click = False
            self.drag_start = None
            self.selection = None # 拖动选框清空
            self.drag_type = None


        elif event == cv2.EVENT_RBUTTONUP:  # 松开右键
            if self.drag_start and self.selection and self.drag_type == 'place':
                self.track_window = self.selection # 最终选框
                with self.lock:
                    default_offset = [0.0, 0.0]
                    self.action_list.append(['place', default_offset,self.track_window])
            #重置状态
            self.mouse_click = False
            self.selection = None
            self.track_window = None
            self.drag_start = None

    def _init_parameters(self):
        self.heart = None
        self.enter = False
        self.start_transport = False
        self.enable_transport = False
        self.sync = None
        self.start_get_roi = False
        self.rgb_sub = None
        self.depth_sub = None
        self.info_sub = None
        self.depth_info_sub = None
        self.white_area_center = None
        self.roi = None
        self.plane = None
        self.extristric = None
        self.corners = None
        self.intrinsic = None
        self.distortion = None
        self.action_list = []
        self.target = []
        self.start_stamp = time.time()

        self.mouse_click = False
        self.selection = None
        self.track_window = None
        self.drag_start = None
        self.drag_type = None

    def init_process(self):
        self.timer.cancel()

        threading.Thread(target=self.main, daemon=True).start()
        threading.Thread(target=self.transport_thread, daemon=True).start()

        if self.get_parameter('start').value:
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            req = SetBool.Request()
            req.data = True 
            res = SetBool.Response()
            self.enable_transport_srv_callback(req, res)

        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def go_home(self, interrupt=True, back=True):
        if interrupt:
            set_servo_position(self.joints_pub, 0.5, ((10, 200), ))
            time.sleep(0.5)
        
        joint_angle = [500, 520, 185, 50, 500]
        
        msg = set_joint_value_target(joint_angle)
        endpoint = self.send_request(self.set_joint_value_target_client, msg)
        pose_t, pose_r = endpoint.pose.position, endpoint.pose.orientation
        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        set_servo_position(self.joints_pub, 1, ((2, joint_angle[1]), (3, joint_angle[2]), (4, joint_angle[3]), (5, 500)))
        time.sleep(1)
        
        if back:
            set_servo_position(self.joints_pub, 1, ((1, joint_angle[0]), ))
            time.sleep(1.5)

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "enter object transport")
        with self.lock:
            self._init_parameters()
            self.heart = Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # Heartbeat package 心跳包
            cv2.namedWindow('image', 1)
            cv2.setMouseCallback('image', self.onmouse)

            if self.sync is None:
                self.rgb_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb0/image_raw')
                self.info_sub = message_filters.Subscriber(self, CameraInfo, '/depth_cam/rgb0/camera_info')
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [self.rgb_sub,  self.info_sub],   # 需要同步的订阅器列表
                     3,                               # 队列大小（缓存消息数）
                     0.2                              # 时间容差（秒）
                     )
                self.sync.registerCallback(self.multi_callback)

            self.enter = True
            self.start_get_roi = True
        
        self.go_home()
        response.success = True
        response.message = "start"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "exit  object transport")
        with self.lock:
            self._init_parameters()
            if self.sync is not None:
                self.sync.disconnect(self.multi_callback)
                self.sync = None
            pick_and_place.interrupt()
        response.success = True
        response.message = "start"
        return response

    def enable_transport_srv_callback(self, request, response):
        with self.lock:
            if request.data:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'enable  object transport')
                self.enable_transport = True
            else:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'exit  object transport')
                pick_and_place.interrupt()
                self.enable_transport = False
        response.success = True
        response.message = "start"
        return response

    def set_pick_position_srv_callback(self, request, response):
        with self.lock:
            # self.get_logger().info('\033[1;32mset pick: %s\033[0m' % request)
            self.action_list.append(['pick', request.box])
        response.success = True
        response.message = "start"
        return response

    def record_position_srv_callback(self, request, response):
        with self.lock:
            # self.get_logger().info('\033[1;32mrecord: %s\033[0m' % request)
            self.action_list.append(['record', request.label, request.box])
        response.success = True
        response.message = "start"
        return response

    def set_place_position_srv_callback(self, request, response):
        with self.lock:
            # self.get_logger().info('\033[1;32mset place: %s\033[0m' % request)
            if request.label and not request.box:
                self.action_list.append(['restore', request.label])
            else:
                self.action_list.append(['place', request.offset, request.box])
        response.success = True
        response.message = "start"
        return response

    def get_roi(self):
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # Convert to numpy array. 转换为 numpy 数组
            corners = np.array(config['corners']).reshape(-1, 3)
            with self.lock:
                self.extristric = np.array(config['extristric'])
                self.white_area_center = np.array(config['white_area_pose_world'])
                self.plane = config['plane']
                self.corners = np.array(config['corners'])

        while self.intrinsic is None or self.distortion is None:
            self.get_logger().info("waiting for camera info")
            time.sleep(0.1)

        with self.lock:
            tvec = self.extristric[:1]  # Take the first row. 取第一行
            rmat = self.extristric[1:]  # Take the last three rows. 取后面三行

            tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03)
            # self.get_logger().info(f'corners: {corners}')
            imgpts, jac = cv2.projectPoints(
                corners[:-1], 
                np.array(rmat), 
                np.array(tvec), 
                self.intrinsic, 
                None   # 如果用校准后的图像，可传 None self.distortion
                )
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Crop RIO region 裁切出ROI区域
            x_min = min(imgpts, key=lambda p: p[0])[0] # The minimum value of the X-axis. x轴最小值
            x_max = max(imgpts, key=lambda p: p[0])[0] # The maximum value of the X-axis. x轴最大值
            y_min = min(imgpts, key=lambda p: p[1])[1] # The minimum value of the Y-axis. y轴最小值
            y_max = max(imgpts, key=lambda p: p[1])[1] # The maximum value of the Y-axis. y轴最大值
            roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
            self.roi = roi

    def cal_grap_point(self, mask, x, y, box, edge_index):
        # Calculate the direction vector of the edge. 计算边的方向向量
        edge_vector = box[(edge_index + 1) % 4] - box[edge_index]
        edge_vector = edge_vector / np.linalg.norm(edge_vector)

        # Calculate direction vector perpendicular to the edge. 计算垂直于边的方向向量
        perpendicular_vector = np.array([-edge_vector[1], edge_vector[0]])

        # Define a long line starting from the centroid. 定义一条从质心出发的长线
        line_length = 1000  # Length of the line. 线的长度
        line_start = (x - int(perpendicular_vector[0] * line_length), y - int(perpendicular_vector[1] * line_length))
        line_end = (x + int(perpendicular_vector[0] * line_length), y + int(perpendicular_vector[1] * line_length))

        # Draw the line on the ROI image. 在ROI图像上绘制这条线
        line_image = np.zeros_like(mask)
        cv2.line(line_image, line_start, line_end, 255, 1, cv2.LINE_AA)

        # Find the intersection between the line and the ROI. 找到线与ROI的交点
        intersection_image = cv2.bitwise_and(mask, line_image)
        intersection_contours, _ = cv2.findContours(intersection_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaMaxContour, area_max = common.get_area_max_contour(intersection_contours)
        if areaMaxContour is not None:
            rect = cv2.minAreaRect(areaMaxContour)  # Obtain the minimum bounding rectangle. 获取最小外接矩形
            center = [rect[0][0], rect[0][1]]
            object_width = max(rect[1])
            return [center, object_width]
        else:
            return False

    def get_object_pixel_position(self, image, roi):
        image_w, image_h = image.shape[1], image.shape[0]
        # roi[x, y, x, y]
        everything_results = self.predictor(image)
        # Prompt inference
        results = prompt(everything_results, bboxes=[roi], log=self.get_logger())
        
        # Get mask from results and ensure it matches original image size
        result = results[0]
        mask = result.masks.data
        
        # Use YOLO's scale_masks function to properly scale mask to original image size
        if mask.shape[1:] != result.orig_shape:
            mask = scale_masks(mask[None], result.orig_shape)[0]

        mask = mask.cpu().numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:  # Might be (1, H, W), need to remove the first dimension. 可能是 (1, H, W) 需要去掉第一维
            mask = mask[0]

        mask = (mask * 255).astype(np.uint8)

        # cv2.imshow('mask', mask)
        M = cv2.moments(mask)
        annotated_frame = results[0].plot()
        # cv2.imshow("YOLO Inference", annotated_frame)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            grasp_point = (cx, cy)
            cv2.circle(image, grasp_point, 5, (255, 0, 0), -1)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            areaMaxContour, area_max = common.get_area_max_contour(contours)
            if areaMaxContour is not None:
                rect = cv2.minAreaRect(areaMaxContour)  # Obtain the minimum bounding rectangle 获取最小外接矩形
                # In version 4.5, w is defined as the edge that first aligns with the x-axis under clockwise rotation; angle is the clockwise rotation from the x-axis, ranging in (0, 90].
                rect_center, (width, height), angle = rect

                bbox = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(image, [bbox], -1, (0, 255, 255), 2, cv2.LINE_AA)  # Draw rectangle contour. 绘制矩形轮廓
                
                cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2, 1)
                # cv2.imshow('img', image)
                # cv2.waitKey(1)
                target_points_data = ["target", 0, rect_center, (width, height), angle]
                return grasp_point, target_points_data
        return False

    def get_object_world_position(self, position, intrinsic, extristric, white_area_center, height=0.025):
        tvec = extristric[:1].reshape((3, 1))  # 取第1行并塑形为 (3, 1)
        rmat = extristric[1:]                  # 取后3行，形状为 (3, 3)   
        stacked_array = np.column_stack((rmat, tvec))  
        projection_matrix = np.row_stack((stacked_array, np.array([[0, 0, 0, 1]])))

        # projection_matrix = np.row_stack((np.column_stack((extristric[1], extristric[0])), np.array([[0, 0, 0, 1]])))

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
        return position, projection_matrix

    def main(self):
        while True:
            if self.enter:
                try:
                    bgr_image, camera_info = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                with self.lock:
                    # 1. 提取内参矩阵
                    self.intrinsic = np.matrix(camera_info.k).reshape(1, -1, 3)
                    # 2. 提取畸变系数
                    self.distortion = np.array(camera_info.d)
                    if self.start_get_roi:
                        self.get_roi()
                        self.start_get_roi = False
                max_dist = 350
                h, w, _ = bgr_image.shape

                if self.enable_transport:
                    with self.lock:
                        if self.action_list:
                            box_info = self.action_list[0]
                            if box_info[0] == 'pick':
                                if not self.target:
                                    grasp_point_pixel, object_angle = self.get_object_pixel_position(bgr_image, box_info[1]) # xyxy
                                    if grasp_point_pixel:
                                        world_position, _ = self.get_object_world_position(grasp_point_pixel, self.intrinsic, self.extristric, self.white_area_center, height=0.025)
                                        yaw = math.degrees(math.atan2(world_position[1], world_position[0]))
                                        if world_position[0] < 0:
                                            if world_position[1] < 0:
                                                yaw += 180
                                            else:
                                                yaw -= 180

                                        # yaw += object_angle
                                        points = []   #传入空列表
                                        gripper_size = self.gripper_pixel_size
                                        optimal_result = calculate_grasp_yaw.calculate_gripper_yaw_angle(object_angle, points, gripper_size, yaw)
                                        if optimal_result is None:
                                            yaw = yaw + object_angle[-1]
                                        else:
                                            yaw, _, _ = optimal_result


                                        # self.get_logger().info(f'[PICK] 最终抓取角度: {yaw} 度')
                                        yaw_servo_value = 500 + int(yaw / 240 * 1000)

                                        gripper_angle = 540  # 固定的爪子张开角度

                                        self.target = [world_position, grasp_point_pixel, yaw_servo_value, gripper_angle, box_info[0]]
                                        # self.get_logger().info(f'target {self.target}')
                                        self.start_stamp = time.time()

                                else:
                                    if time.time() - self.start_stamp > 2:
                                        self.start_transport = True        
                                        self.enable_transport = False
                                    cv2.circle(bgr_image, (int(self.target[1][0]), int(self.target[1][1])), 10, (255, 0, 0), -1)

                                #使用cv2.rectangle将大模型返回的最表绘制出框
                                cv2.rectangle(bgr_image, (box_info[-1][0], box_info[-1][1]), (box_info[-1][2], box_info[-1][3]), (0, 255, 0), 2, 1)
                            elif box_info[0] == 'place':
                                if not self.target:
                                    grasp_point_pixel, object_angle = self.get_object_pixel_position(bgr_image, box_info[-1]) # xyxy
                                    # self.get_logger().info(str(object_info))
                                    if grasp_point_pixel:
                                        world_position, _ = self.get_object_world_position(grasp_point_pixel, self.intrinsic, self.extristric, self.white_area_center, height=0.010)
                                        #加上偏移量 例如 放在盘子左边的情况
                                        world_position[0] += box_info[-2][0]
                                        world_position[1] += box_info[-2][1]
                                        yaw = math.degrees(math.atan2(world_position[1], world_position[0]))
                                        if world_position[0] < 0:
                                            if world_position[1] < 0:
                                                yaw += 180
                                            else:
                                                yaw -= 180
                                         
                                        points = []   #传入空列表
                                        gripper_size = self.gripper_pixel_size
                                        optimal_result = calculate_grasp_yaw.calculate_gripper_yaw_angle(object_angle, points, gripper_size, yaw)
                                        if optimal_result is None:
                                            yaw = yaw + object_angle[-1]
                                        else:
                                            yaw, _, _ = optimal_result

                                        yaw_servo_value = 500 + int(yaw / 240 * 1000)
                                        gripper_angle = 500
                                        self.target = [world_position, grasp_point_pixel, yaw_servo_value, gripper_angle, box_info[0]] #把 'place' 动作的标签加到 self.target 列表的末尾

                                        # self.target[0][0] += box_info[-2][0]
                                        # self.target[0][1] += box_info[-2][1] 
                                        if box_info[-2][0] != 0 or box_info[-2][1] != 0:                                           
                                            self.target[0][2] = 0.015
                                        else:
                                            self.target[0][2] += 0.015
                                        # self.get_logger().info(f'target {self.target}')
                                        self.start_stamp = time.time()
                                else:
                                    if time.time() - self.start_stamp > 2:
                                        self.start_transport = True       #暂时停止夹取 
                                        self.enable_transport = False
                                    cv2.circle(bgr_image, (int(self.target[1][0]), int(self.target[1][1])), 10, (255, 0, 0), -1)
                                cv2.rectangle(bgr_image, (box_info[-1][0], box_info[-1][1]), (box_info[-1][2], box_info[-1][3]), (0, 255, 0), 2, 1)
                            elif box_info[0] == 'record':
                                grasp_point_pixel, object_angle = self.get_object_pixel_position(bgr_image, box_info[-1]) # xyxy
                                if grasp_point_pixel:
                                    world_position, _ = self.get_object_world_position(grasp_point_pixel, self.intrinsic, self.extristric, self.white_area_center, height=0.015)
                                    yaw = math.degrees(math.atan2(world_position[1], world_position[0]))
                                    if world_position[0] < 0:
                                        if world_position[1] < 0:
                                            yaw += 180
                                        else:
                                            yaw -= 180
                                    # yaw += object_angle 

                                    points = []   #传入空列表
                                    gripper_size = self.gripper_pixel_size
                                    optimal_result = calculate_grasp_yaw.calculate_gripper_yaw_angle(object_angle, points, gripper_size, yaw)
                                    if optimal_result is None:
                                        yaw = yaw + object_angle[-1]
                                    else:
                                        yaw, _, _ = optimal_result

                                    yaw_servo_value = 500 + int(yaw / 240 * 1000)
                                    gripper_angle = 500
                                    self.target = [world_position, grasp_point_pixel, yaw_servo_value, gripper_angle, box_info[0]]

                                    cv2.circle(bgr_image, (int(self.target[1][0]), int(self.target[1][1])), 10, (255, 0, 0), -1)
                                    self.record_position.append([box_info[1], self.target])
                                    del self.action_list[0]
                                    if not self.action_list:
                                        speech.play_audio(record_finish_audio_path)  #暂时注释
                                        self.enable_transport = False
                                        self.target = []
                                cv2.rectangle(bgr_image, (box_info[-1][0], box_info[-1][1]), (box_info[-1][2], box_info[-1][3]), (0, 255, 0), 2, 1)
                            elif box_info[0] == 'restore':
                               for i in self.record_position:
                                   if box_info[1].lower() in i[0].lower():
                                       self.target = copy.deepcopy(i[1])
                                       self.target[-1] = 'restore'
                                       
                                       self.start_transport = True
                                       self.enable_transport = False
                                       break
                        else:
                            msg = Bool()
                            msg.data = True
                            self.transport_finished_pub.publish(msg)
                if self.track_window:  # 绘制最终确定的选框 (红色)
                    cv2.rectangle(bgr_image, (self.track_window[0], self.track_window[1]),
                                  (self.track_window[2], self.track_window[3]), (0, 0, 255), 2)
                elif self.selection:  # 绘制正在拖动的选框 (黄色)
                    cv2.rectangle(bgr_image, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]),
                                  (0, 255, 255), 2)

                self.fps.update()
                self.fps.show_fps(bgr_image)
                # result_image = np.concatenate([bgr_image[40:440, ], depth_color_map], axis=1)
                cv2.imshow('image', bgr_image)
                cv2.waitKey(1)
                if not self.set_above:
                    cv2.moveWindow('image', 1920 - 640*2, 0)
                    os.system("wmctrl -r image -b add,above")
                    self.set_above = True
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()

    def transport_thread(self):
        while True:
            if self.start_transport:
                pose = []
                config_data = common.get_yaml_data(os.path.join(self.config_path, self.calibration_file))
                offset = tuple(config_data['kinematics']['offset'])
                scale = tuple(config_data['kinematics']['scale'])
                p = copy.deepcopy(self.target)
                for i in range(3):
                    p[0][i] = p[0][i] * scale[i]
                    p[0][i] = p[0][i] + offset[i]
                # self.get_logger().info(f'pick_and_place: {p}')
                if self.target[-1] == 'pick':
                    finish = pick_and_place.pick_without_back(p[0], 80, p[2], 500, 0.015, self.joints_pub, self.kinematics_client)
                    if not finish:
                        self.go_home()
                    else:
                        self.go_home(False)
                elif self.target[-1] == 'place':
                    finish = pick_and_place.place(p[0], 80, p[2], 400, self.joints_pub, self.kinematics_client)
                    if not finish:
                        self.go_home()
                    else:
                        self.go_home(False)
                elif self.target[-1] == 'restore':
                    p[0][-1] -= 0.015
                    finish = pick_and_place.place(p[0], 80, p[2], 400, self.joints_pub, self.kinematics_client)
                    if not finish:
                        self.go_home()
                    else:
                        self.go_home(False)
                with self.lock:
                    self.target = []
                    del self.action_list[0]
                self.enable_transport = True
                self.start_transport = False
            else:
                time.sleep(0.1)

    def multi_callback(self, ros_rgb_image,  camera_info):
        cv_image = self.bridge.imgmsg_to_cv2(ros_rgb_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # # If the queue is full, discard the oldest image. 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
            # # Put the image into the queue. 将图像放入队列
        self.image_queue.put((bgr_image, camera_info))

def main():
    node = ObjectTransport('object_transport')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
