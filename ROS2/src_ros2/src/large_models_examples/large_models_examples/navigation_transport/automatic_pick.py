#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/11/18
# @author:aiden
# Tracking and picking(追踪拾取)
import os
import ast
import cv2
import time
import math
import torch
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
from ultralytics.models.fastsam import FastSAMPredictor
from interfaces.srv import SetPose2D, SetPoint, SetBox
from servo_controller.bus_servo_control import set_servo_position
from servo_controller.action_group_controller import ActionGroupController
from ultralytics.utils.ops import scale_masks

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

# display_size = [int(640*6/4), int(480*6/4)]
class AutomaticPickNode(Node):
    config_path = '/home/ubuntu/ros2_ws/src/large_models_examples/config/automatic_pick_roi.yaml'

    lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
    
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        # Color recognition(颜色识别)
        self.depth_cam_type = os.environ['DEPTH_CAMERA_TYPE']
        if self.depth_cam_type  == 'aurora':
            self.image_proc_size = (320, 200)
        elif self.depth_cam_type  == 'usb_cam' or self.depth_cam_type  == 'ascamera':
            self.image_proc_size = (320, 240)
        self.color_picker_point = []
        self.threshold = 0.5
        self.color_picker = None
        self.min_color = []
        self.max_color = []
        self.box = []
        
        self.mouse_click = False
        self.selection = None  # Tracking area that follows the mouse in real time(实时跟踪鼠标的跟踪区域)
        self.track_window = None  # Region where the object to be detected is located(要检测的物体所在区域)
        self.drag_start = None  # Flag indicating whether mouse dragging has started(标记，是否开始拖动鼠标)
        self.start_circle = True
        self.start_click = False
        self.pick_finish = False
        self.place_finish = False

        self.running = True
        self.detect_count = 0
        self.start_pick = False
        self.start_place = False
        self.linear_base_speed = 0.007
        self.angular_base_speed = 0.03

        self.yaw_pid = PID(P=0.015, I=0, D=0.000)
        # self.linear_pid = PID(P=0, I=0, D=0)
        self.linear_pid = PID(P=0.0018, I=0, D=0)
        # self.angular_pid = PID(P=0, I=0, D=0)
        self.angular_pid = PID(P=0.003, I=0, D=0)

        self.linear_speed = 0
        self.angular_speed = 0
        self.yaw_angle = 90

        self.pick_stop_x = 320
        self.pick_stop_y = 388
        self.place_stop_x = 320
        self.place_stop_y = 388
        self.stop = True

        self.d_y = 10
        self.d_x = 10

        self.pick = False
        self.place = False

        self.status = "approach"
        self.count_stop = 0
        self.count_turn = 0

        self.declare_parameter('status', 'start')
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        
        self.start = self.get_parameter('start').value
        self.enable_display = self.get_parameter('enable_display').value
        self.debug = self.get_parameter('debug').value
        self.image_name = 'image'
        cv2.namedWindow(self.image_name, 1)
        cv2.setMouseCallback(self.image_name, self.onmouse)
        # cv2.namedWindow('image', 0)
        # cv2.moveWindow('image', 1920 - display_size[0], 0)
        # cv2.waitKey(10)
        # os.system("wmctrl -r image -b add,above")  
        code_path = os.path.abspath(os.path.split(os.path.realpath(__file__))[0])
        overrides = dict(conf=0.4, task="segment", mode="predict", model=os.path.join(os.path.dirname(code_path), 'resources/models', "FastSAM-x.pt"), save=False, imgsz=(480, 640))
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor(np.zeros((640, 480, 3), dtype=np.uint8))
        self.language = os.environ['ASR_LANGUAGE']
        self.machine_type = os.environ['MACHINE_TYPE']

        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.image_pub = self.create_publisher(Image, '~/image_result', 1)

        image_topic = '/depth_cam/rgb0'#self.get_parameter('depth_camera/camera_name').value
        self.create_subscription(Image, image_topic + '/image_raw', self.image_callback, 1)
        
        self.create_service(Trigger, '~/pick', self.start_pick_callback)
        self.create_service(Trigger, '~/place', self.start_place_callback) 
        self.create_service(SetPoint, '~/set_target_color', self.set_target_color_srv_callback)
        self.create_service(SetBox, '~/set_box', self.set_box_srv_callback)

        self.controller = ActionGroupController(self.create_publisher(ServosPosition, 'servo_controller', 1), '/home/ubuntu/software/arm_pc/ActionGroups')
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        
        self.action_finish_pub = self.create_publisher(Bool, '~/action_finish', 1)
        
        self.mecanum_pub.publish(Twist())
        set_servo_position(self.joints_pub, 2, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 200)))
        time.sleep(2)
        # msg = Trigger.Request()
        # self.start_pick_callback(msg, Trigger.Response())
        if self.debug == 'pick':
            self.controller.run_action('navigation_pick_debug')
            time.sleep(5)
            set_servo_position(self.joints_pub, 2, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 200)))
            time.sleep(2)
            msg = Trigger.Request()
            self.start_pick_callback(msg, Trigger.Response())
        elif self.debug == 'place':
            self.controller.run_action('navigation_place_debug')
            time.sleep(5)
            set_servo_position(self.joints_pub, 2, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 200)))
            time.sleep(2)
            msg = Trigger.Request()
            self.start_place_callback(msg, Trigger.Response())

        threading.Thread(target=self.action_thread, daemon=True).start()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        return response


    # Mouse click event callback function(鼠标点击事件回调函数)
    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Press the left mouse button(鼠标左键按下)
            self.mouse_click = True
            self.drag_start = (x, y)  # Starting position of the mouse(鼠标起始位置)
            self.track_window = None
            self.selection = None
            self.start_circle = True
            self.start_click = False
        if self.drag_start:  # Whether dragging has started; records the mouse position(是否开始拖动鼠标，记录鼠标位置)
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:  # Mouse left button release(鼠标左键松开)
            if not self.debug:
                if not self.pick_finish:
                    msg = Trigger.Request()
                    self.start_pick_callback(msg, Trigger.Response())
                elif not self.place_finish:
                    msg = Trigger.Request()
                    self.start_place_callback(msg, Trigger.Response())
                else:
                    msg = Trigger.Request()
                    self.start_pick_callback(msg, Trigger.Response())
            self.mouse_click = False
            self.drag_start = None
            self.track_window = self.selection
            self.selection = None
        if event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_click = False
            self.selection = None  # Tracking area that follows the mouse in real time(实时跟踪鼠标的跟踪区域)
            self.track_window = None  # Region where the object to be detected is located(要检测的物体所在区域)
            self.drag_start = None  # Flag indicating whether mouse dragging has started(标记，是否开始拖动鼠标)
            self.start_circle = True
            self.start_click = False
            self.start_pick = False
            self.pick = False
            self.place = False
            self.max_color, self.min_color = [], []

    def set_box_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_box')
        self.box = [request.x_min, request.y_min, request.x_max, request.y_max]
        self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_box"
        return response

    def set_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_target_color')
        x, y = request.data.x, request.data.y
        if x == -1 and y == -1:
            self.min_color, self.max_color = [], []
            self.color_picker = None
        else:
            self.min_color, self.max_color = [], []
            self.color_picker = ColorPicker(request.data, 10)
        self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_target_color"
        return response

    def start_pick_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start pick")
        
        self.place_finish = False
        self.linear_speed = 0
        self.angular_speed = 0
        self.yaw_angle = 90

        param = self.get_parameter('pick_stop_pixel_coordinate').value
        self.get_logger().info('\033[1;32mget pick stop pixel coordinate: %s\033[0m' % str(param))
        self.pick_stop_x = param[0]
        self.pick_stop_y = param[1]
        self.stop = True

        self.d_y = 10
        self.d_x = 10

        self.pick = False
        self.place = False

        self.status = "approach"
        self.count_stop = 0
        self.count_turn = 0

        self.linear_pid.clear()
        self.angular_pid.clear()
        self.start_pick = True

        response.success = True
        response.message = "start_pick"
        return response 


    def start_place_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start place")
        
        self.pick_finish = False
        self.linear_speed = 0
        self.angular_speed = 0
        self.d_y = 30
        self.d_x = 30
        
        param = self.get_parameter('place_stop_pixel_coordinate').value
        self.get_logger().info('\033[1;32mget place stop pixel coordinate: %s\033[0m' % str(param))
        self.place_stop_x = param[0]
        self.place_stop_y = param[1]
        self.stop = True
        self.pick = False
        self.place = False

        self.linear_pid.clear()
        self.angular_pid.clear()
        self.start_place = True

        response.success = True
        response.message = "start_place"
        return response 

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()


    def color_detect(self, img):
        img_h, img_w = img.shape[:2]
        frame_resize = cv2.resize(img, self.image_proc_size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)  # Convert image to LAB space(将图像转换到LAB空间)
        frame_mask = cv2.inRange(frame_lab, tuple(self.min_color), tuple(self.max_color))  # Perform bitwise operation on the original image and the mask(对原图像和掩模进行位运算)

        eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # Erode(腐蚀)
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # Dilate(膨胀)

        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Find contours(找出轮廓)
        # cv2.imshow('image', dilated)
        center_x, center_y, angle = -1, -1, -1
        if len(contours) != 0:
            areaMaxContour, area_max = common.get_area_max_contour(contours, 10)  # Find the largest contour(找出最大轮廓)
            if areaMaxContour is not None:
                if 10 < area_max:  # The maximum area has been found(有找到最大面积)
                    rect = cv2.minAreaRect(areaMaxContour)  # The minimum bounding rectangle(最小外接矩形)
                    angle = rect[2]
                    box = np.intp(cv2.boxPoints(rect))  # The four corner points of the minimum bounding rectangle(最小外接矩形的四个顶点)
                    for j in range(4):
                        box[j, 0] = int(common.val_map(box[j, 0], 0, self.image_proc_size[0], 0, img_w))
                        box[j, 1] = int(common.val_map(box[j, 1], 0, self.image_proc_size[1], 0, img_h))

                    cv2.drawContours(img, [box], -1, (0, 255, 255), 2)  # Draw the rectangle composed of the four points(画出四个点组成的矩形)
                    # Obtain the diagonal points of the rectangle(获取矩形的对角点)
                    ptime_start_x, ptime_start_y = box[0, 0], box[0, 1]
                    pt3_x, pt3_y = box[2, 0], box[2, 1]
                    radius = abs(ptime_start_x - pt3_x)
                    center_x, center_y = int((ptime_start_x + pt3_x) / 2), int((ptime_start_y + pt3_y) / 2)  # Center point(中心点)
                    cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)  # Draw the center point(画出中心点)

        return center_x, center_y, angle

    def segment_handle(self, image, box, factor):
        everything_results = self.predictor(image)
        results = prompt(everything_results, bboxes=[box])
        # annotated_frame = results[0].plot()
        
        # Get mask from results and ensure it matches original image size
        result = results[0]
        mask = result.masks.data
        
        # Use YOLO's scale_masks function to properly scale mask to original image size
        if mask.shape[1:] != result.orig_shape:
            mask = scale_masks(mask[None], result.orig_shape)[0]
        
        mask = mask.cpu().numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:  # It might be (1, H, W), remove the first dimension(可能是 (1, H, W) 需要去掉第一维)
            mask = mask[0]
        
        mask = (mask * 255).astype(np.uint8)
        # cv2.imshow('mask', mask)
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2, 1)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        areaMaxContour, area_max = common.get_area_max_contour(contours, 10)
        center_x, center_y, angle = -1, -1, -1
        if areaMaxContour is not None:
            if 10 < area_max:
                rect = cv2.minAreaRect(areaMaxContour)  # Obtain the minimum bounding rectangle(获取最小外接矩形)
                #In version 4.5, the edge that first overlaps with the X-axis during a clockwise rotation is defined as w. The angle is the clockwise rotation angle of the X-axis, with a value range of (0, 90].
                # (4.5版本定义为，x轴顺时针旋转最先重合的边为w，angle为x轴顺时针旋转的角度，angle取值为(0,90])
                angle = rect[2]
                box = np.intp(cv2.boxPoints(rect))  # The four corner points of the minimum bounding rectangle(最小外接矩形的四个顶点)

                cv2.drawContours(image, [box], -1, (0, 255, 255), 2)  # Draw the rectangle composed of the four points(画出四个点组成的矩形)
                # Obtain the diagonal points of the rectangle(获取矩形的对角点)
                ptime_start_x, ptime_start_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                radius = abs(ptime_start_x - pt3_x)
                center_x, center_y = int((ptime_start_x + pt3_x) / 2), int((ptime_start_y + pt3_y) / 2)  # Center point(中心点)
                center_y = int(center_y - rect[1][1] / 2 + rect[1][1] * factor)
                cv2.circle(image, (center_x, center_y), 5, (0, 255, 255), -1)  # Draw the center point(画出中心点)
                h, w = image.shape[:2]
                center_x /= w
                center_y /= h
        # cv2.imshow("image_draw", image)
        # cv2.waitKey(10000)
        return float(center_x), float(center_y)

    def action_thread(self):
        while True:
            if self.pick:
                self.min_color, self.max_color = [], []
                self.start_pick = False
                self.mecanum_pub.publish(Twist())

                time.sleep(0.5)
                set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 100), (3, 450), (4, 230), (5, 500), (10, 200)))
                time.sleep(2)
                set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 100), (3, 450), (4, 230), (5, 500), (10, 200)))
                time.sleep(0.5)
                set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 100), (3, 450), (4, 230), (5, 500), (10, 560)))
                time.sleep(0.5)
                set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 100), (3, 450), (4, 230), (5, 500), (10, 560)))
                time.sleep(0.3)
                set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 560)))
                time.sleep(1.5)
                set_servo_position(self.joints_pub, 0.3, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 560)))
                time.sleep(0.3)

                self.pick = False
                self.get_logger().info('pick finish')
                msg = Bool()
                msg.data = True
                self.action_finish_pub.publish(msg)
                self.pick_finish = True
                # set_servo_position(self.joints_pub, 0.3, ((10, 200),))
                # time.sleep(0.3)
                # self.start_click = False
                # self.start_circle = True
                self.place = False
            elif self.place:
                self.min_color, self.max_color = [], []
                self.start_place = False
                self.mecanum_pub.publish(Twist())
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 200), (3, 370), (4, 260), (5, 500), (10, 560)))
                time.sleep(1.5)
                set_servo_position(self.joints_pub, 0.3, ((1, 500), (2, 180), (3, 350), (4, 297), (5, 500), (10, 560)))
                time.sleep(0.3)
                set_servo_position(self.joints_pub, 0.5, ((1, 500), (2, 180), (3, 350), (4, 297), (5, 500), (10, 200)))
                time.sleep(0.5)
                set_servo_position(self.joints_pub, 0.3, ((1, 500), (2, 180), (3, 350), (4, 297), (5, 500), (10, 200)))
                time.sleep(0.3)
                set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 200)))
                time.sleep(1.5)
                set_servo_position(self.joints_pub, 0.3, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 200)))
                time.sleep(0.3)
                self.get_logger().info('place finish')
                self.place = False
                msg = Bool()
                msg.data = True
                self.action_finish_pub.publish(msg)
                # self.start_click = False
                # self.start_circle = True
                self.place_finish = True
            else:
                time.sleep(0.01)

    def pick_handle(self, image):
        twist = Twist()

        if not self.pick or self.debug == 'pick':
            object_center_x, object_center_y, object_angle = self.color_detect(image)  # Obtain the center and angle of the object color(获取物体颜色的中心和角度)
            if self.debug == 'pick':
                self.detect_count += 1
                if self.detect_count > 10:
                    self.detect_count = 0
                    self.pick_stop_y = object_center_y
                    self.pick_stop_x = object_center_x
                    data = common.get_yaml_data(self.config_path)
                    data['/**']['ros__parameters']['pick_stop_pixel_coordinate'] = [self.pick_stop_x, self.pick_stop_y]
                    common.save_yaml_data(data, self.config_path)
                    self.debug = False
                self.get_logger().info('x_y: ' + str([object_center_x, object_center_y]))  # Print the pixel of the current object's center(打印当前物体中心的像素)
            elif object_center_x > 0:
                ########Motor PID processing(电机pid处理)#########
                # Use the x and y coordinates of the image's center point as the set value, and the current x and y coordinates as the input(以图像的中心点的x，y坐标作为设定的值，以当前x，y坐标作为输入)#
                self.linear_pid.SetPoint = self.pick_stop_y
                # self.get_logger().info(f'{self.pick_stop_y} {self.d_y} {object_center_y} {object_center_y - self.pick_stop_y}')
                if abs(object_center_y - self.pick_stop_y) <= self.d_y:
                    object_center_y = self.pick_stop_y
                if self.status != "align":
                    self.linear_pid.update(object_center_y)  # Update PID(更新pid)
                    output = self.linear_pid.output
                    tmp = math.copysign(self.linear_base_speed, output) + output
                    # self.get_logger().info(f'{tmp}')
                    self.linear_speed = tmp
                    if tmp > 0.15:
                        self.linear_speed = 0.15
                    if tmp < -0.15:
                        self.linear_speed = -0.15
                    if abs(tmp) <= 0.0075:
                        self.linear_speed = 0

                self.angular_pid.SetPoint = self.pick_stop_x
                if abs(object_center_x - self.pick_stop_x) <= self.d_x:
                    object_center_x = self.pick_stop_x
                if self.status != "align":
                    self.angular_pid.update(object_center_x)  # Update PID(更新pid)
                    output = self.angular_pid.output
                    tmp = math.copysign(self.angular_base_speed, output) + output

                    self.angular_speed = tmp
                    if tmp > 1.2:
                        self.angular_speed = 1.2
                    if tmp < -1.2:
                        self.angular_speed = -1.2
                    if abs(tmp) <= 0.038:
                        self.angular_speed = 0
                if abs(self.linear_speed) == 0 and abs(self.angular_speed) == 0:
                    self.count_turn += 1
                    if self.count_turn > 5:
                        self.count_turn = 5
                        self.status = "align"
                        if self.count_stop < 10:  # If there is no movement detected for 10 consecutive times(连续10次都没在移动)
                            if object_angle < 40: # Do not use 45, because unstable values at 45 may cause repeated movement(不取45，因为如果在45时值的不稳定会导致反复移动)
                                object_angle += 90
                            self.yaw_pid.SetPoint = 90
                            if abs(object_angle - 90) <= 1:
                                object_angle = 90
                            self.yaw_pid.update(object_angle)  # Update PID(更新pid)
                            self.yaw_angle = self.yaw_pid.output
                            if object_angle != 90:
                                if abs(self.yaw_angle) <=0.038:
                                    self.count_stop += 1
                                else:
                                    self.count_stop = 0
                                twist.linear.y = float(-2 * 0.3 * math.sin(self.yaw_angle / 2))
                                twist.angular.z = float(self.yaw_angle)
                            else:
                                self.count_stop += 1
                        elif self.count_stop <= 20:
                            self.d_x = 4
                            self.d_y = 4
                            self.count_stop += 1
                            self.status = "adjust"
                        else:
                            self.count_stop = 0
                            self.pick = True
                else:
                    if self.count_stop >= 10:
                        self.count_stop = 10
                    self.count_turn = 0
                    if self.status != 'align':
                        twist.linear.x = float(self.linear_speed)
                        twist.angular.z = float(self.angular_speed)

        self.mecanum_pub.publish(twist)

        return image


    def place_handle(self, image):
        twist = Twist()
        h, _ = image.shape[:2]
        image_roi = image[:int(h*0.7), :]
        if not self.place or self.debug == 'place':
            object_center_x, object_center_y, object_angle = self.color_detect(image_roi)  # Obtain the center and angle of the object color(获取物体颜色的中心和角度)
            if self.debug == 'place':
                self.detect_count += 1
                if self.detect_count > 10:
                    self.detect_count = 0
                    self.place_stop_y = object_center_y
                    self.place_stop_x = object_center_x
                    data = common.get_yaml_data(self.config_path)
                    data['/**']['ros__parameters']['place_stop_pixel_coordinate'] = [self.place_stop_x, self.place_stop_y]
                    common.save_yaml_data(data, self.config_path)
                    self.debug = False
                self.get_logger().info('x_y: ' + str([object_center_x, object_center_y]))  # Print the pixel of the current object's center(打印当前物体中心的像素)
            elif object_center_x > 0:
                ########Motor PID processing(电机pid处理)#########
                # Use the x and y coordinates of the image's center point as the set value, and the current x and y coordinates as the input(以图像的中心点的x，y坐标作为设定的值，以当前x，y坐标作为输入)#
                self.linear_pid.SetPoint = self.place_stop_y
                if abs(object_center_y - self.place_stop_y) <= self.d_y:
                    object_center_y = self.place_stop_y
                self.linear_pid.update(object_center_y)  # Update PID(更新pid)
                output = self.linear_pid.output
                tmp = math.copysign(self.linear_base_speed, output) + output

                self.linear_speed = tmp
                if tmp > 0.15:
                    self.linear_speed = 0.15
                if tmp < -0.15:
                    self.linear_speed = -0.15
                if abs(tmp) <= 0.0075:
                    self.linear_speed = 0

                self.angular_pid.SetPoint = self.place_stop_x
                if abs(object_center_x - self.place_stop_x) <= self.d_x:
                    object_center_x = self.place_stop_x

                self.angular_pid.update(object_center_x)  # Update PID(更新pid)
                output = self.angular_pid.output
                tmp = math.copysign(self.angular_base_speed, output) + output

                self.angular_speed = tmp
                if tmp > 1.2:
                    self.angular_speed = 1.2
                if tmp < -1.2:
                    self.angular_speed = -1.2
                if abs(tmp) <= 0.035:
                    self.angular_speed = 0

                if abs(self.linear_speed) == 0 and abs(self.angular_speed) == 0:
                    self.place = True
                else:
                    twist.linear.x = float(self.linear_speed)
                    twist.angular.z = float(self.angular_speed)

        self.mecanum_pub.publish(twist)

        return image


    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            #  If the queue is full, remove the oldest image(如果队列已满，丢弃最旧的图像)
            self.image_queue.get()
            # Put the image into the queue(将图像放入队列)
        self.image_queue.put(rgb_image)

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
            if self.start_circle:
                # Draw a box with the mouse to specify the region(用鼠标拖拽一个框来指定区域)
                if self.track_window:  # Once the tracking window is drawn, highlight the target in real time（跟踪目标的窗口画出后，实时标出跟踪目标）
                    cv2.rectangle(result_image, (self.track_window[0], self.track_window[1]),
                                  (self.track_window[2], self.track_window[3]), (0, 0, 255), 2)
                elif self.selection:  # Display the tracking window in real time as the mouse is dragged（跟踪目标的窗口随鼠标拖动实时显示）
                    cv2.rectangle(result_image, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]),
                                  (0, 255, 255), 2)
                if self.mouse_click:
                    self.start_click = True
                if self.start_click:
                    if not self.mouse_click:
                        self.start_circle = False
                if not self.start_circle:
                    self.box = self.track_window
            # self.get_logger().info('\033[1;32m%s\033[0m' % 'box: ' + str(self.box))
            if self.box:
                if self.start_pick:
                    point = Point()
                    point.x, point.y = self.segment_handle(result_image, self.box, 1 / 2)
                    self.color_picker = ColorPicker(point, 10)
                    self.box = []
                elif self.start_place:
                    point = Point()
                    point.x, point.y = self.segment_handle(result_image, self.box, 1 / 2)
                    self.color_picker = ColorPicker(point, 10)
                    self.box = []
            if self.color_picker is not None:  # Color picker exists(拾取器存在)
                target_color, result_image = self.color_picker(image, result_image)
                if target_color is not None:
                    self.color_picker = None
                    self.min_color = [int(target_color[0][0] - 50 * self.threshold * 2),
                                 int(target_color[0][1] - 50 * self.threshold),
                                 int(target_color[0][2] - 50 * self.threshold)]
                    self.max_color = [int(target_color[0][0] + 50 * self.threshold * 2),
                                 int(target_color[0][1] + 50 * self.threshold),
                                 int(target_color[0][2] + 50 * self.threshold)]
            else:
                if self.max_color and self.min_color:
                    if self.start_pick:
                        self.stop = True
                        result_image = self.pick_handle(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    elif self.start_place:
                        self.stop = True
                        result_image = self.place_handle(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    else:
                        if self.stop:
                            self.stop = False
                            self.mecanum_pub.publish(Twist())
                        result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                    cv2.line(result_image, (self.pick_stop_x, self.pick_stop_y - 10), (self.pick_stop_x, self.pick_stop_y + 10), (0, 255, 255), 2)
                    cv2.line(result_image, (self.pick_stop_x - 10, self.pick_stop_y), (self.pick_stop_x + 10, self.pick_stop_y), (0, 255, 255), 2)

            if self.enable_display:
                cv2.imshow(self.image_name, result_image)
                # cv2.imshow(self.image_name, cv2.resize(result_image, (display_size[0], display_size[1])))  # Display result_image(显示图像)
                cv2.waitKey(1)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "bgr8"))

        set_servo_position(self.joints_pub, 2, ((1, 500), (2, 700), (3, 15), (4, 180), (5, 500), (10, 200)))
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

def main():
    node = AutomaticPickNode('automatic_pick')
    rclpy.spin(node)
    node.destroy_node()
 
if __name__ == "__main__":
    main()

