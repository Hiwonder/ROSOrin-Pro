#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Shape Recognition Upper Computer Interface
用于可视化和控制基于RGB-D相机的物体识别系统

功能特点:
1. 实时显示RGB和深度图像
2. 可视化物体检测结果
3. 显示处理管道的各个步骤
4. 提供系统控制界面
5. 实时参数调节和调试信息
"""

import sys
import cv2
import yaml
import queue
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image as RosImage, CameraInfo
    from std_srvs.srv import Trigger, SetBool
    from interfaces.srv import SetStringList
    from cv_bridge import CvBridge
    import message_filters
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS2 not available, running in demo mode")

class ObjectDetectionVisualizer:
    """物体检测结果可视化类"""
    
    def __init__(self):
        self.colors = {
            'sphere': (0, 255, 0),      # 绿色
            'cuboid': (255, 0, 0),      # 蓝色  
            'cylinder': (0, 0, 255),    # 红色
            'cylinder_horizontal': (255, 255, 0)  # 青色
        }
    
    def draw_detection_results(self, image, objects, show_details=True):
        """在图像上绘制检测结果"""
        annotated_image = image.copy()
        
        for obj_info in objects:
            if len(obj_info) >= 5:
                obj_name = obj_info[0]
                obj_index = obj_info[1]
                position = obj_info[2] if len(obj_info) > 2 else [0, 0, 0]
                bbox_info = obj_info[4] if len(obj_info) > 4 else None
                
                # 获取颜色
                base_name = obj_name.split('_')[0]
                color = self.colors.get(base_name, (128, 128, 128))
                
                if bbox_info and len(bbox_info) >= 4:
                    x, y, w, h = bbox_info[:4]
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                    
                    # 绘制标签
                    label = f"{obj_name}_{obj_index}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(annotated_image, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if show_details and len(bbox_info) >= 7:
                        # 绘制中心点和方向
                        center, width, height, angle = bbox_info[4:8]
                        cv2.circle(annotated_image, (int(center[0]), int(center[1])), 5, color, -1)
                        
                        # 绘制方向指示器
                        length = max(width, height) / 2
                        end_x = int(center[0] + length * np.cos(np.radians(angle)))
                        end_y = int(center[1] + length * np.sin(np.radians(angle)))
                        cv2.arrowedLine(annotated_image, (int(center[0]), int(center[1])), 
                                      (end_x, end_y), color, 2)
        
        return annotated_image

class ROS2ShapeRecognitionNode(Node):
    """ROS2形状识别节点"""
    
    def __init__(self, gui_callback=None):
        super().__init__('shape_recognition_gui_node')
        self.gui_callback = gui_callback
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=5)
        
        # 订阅话题
        self.setup_subscribers()
        
        # 创建服务客户端
        self.setup_service_clients()
        
        # 数据存储
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_result = None
        self.camera_info = None
        self.detection_results = []
        
    def setup_subscribers(self):
        """设置订阅者"""
        # 订阅相机数据
        self.rgb_sub = message_filters.Subscriber(
            self, RosImage, '/ascamera/camera_publisher/rgb0/image')
        self.depth_sub = message_filters.Subscriber(
            self, RosImage, '/ascamera/camera_publisher/depth0/image_raw')
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/ascamera/camera_publisher/rgb0/camera_info')
        
        # 订阅处理结果
        self.result_sub = self.create_subscription(
            RosImage, '/shape_recognition/image_result', 
            self.result_callback, 10)
        
        # 同步相机数据
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], 10, 0.2)
        self.sync.registerCallback(self.camera_callback)
        
    def setup_service_clients(self):
        """设置服务客户端"""
        self.enter_client = self.create_client(Trigger, '/shape_recognition/enter')
        self.exit_client = self.create_client(Trigger, '/shape_recognition/exit')
        self.set_running_client = self.create_client(SetBool, '/shape_recognition/set_running')
        self.set_shape_client = self.create_client(SetStringList, '/shape_recognition/set_shape')
        self.rgb_depth_client = self.create_client(SetBool, '/shape_recognition/rgb_or_depth')
        
    def camera_callback(self, rgb_msg, depth_msg, info_msg):
        """相机数据回调"""
        try:
            # 转换图像
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            self.latest_rgb = rgb_image
            self.latest_depth = depth_image
            self.camera_info = info_msg
            
            # 通知GUI更新
            if self.gui_callback:
                self.gui_callback('camera_update', {
                    'rgb': rgb_image,
                    'depth': depth_image,
                    'info': info_msg
                })
                
        except Exception as e:
            self.get_logger().error(f"Camera callback error: {e}")
    
    def result_callback(self, msg):
        """结果图像回调"""
        try:
            result_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_result = result_image
            
            if self.gui_callback:
                self.gui_callback('result_update', {'result': result_image})
                
        except Exception as e:
            self.get_logger().error(f"Result callback error: {e}")
    
    def call_service(self, client, request):
        """调用服务"""
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.done():
            return future.result()
        return None

class ShapeRecognitionGUI:
    """形状识别上位机主界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ROS2 Shape Recognition Upper Computer")
        self.root.geometry("1400x900")
        
        # 初始化组件
        self.visualizer = ObjectDetectionVisualizer()
        self.ros_node = None
        self.executor = None
        self.ros_thread = None
        
        # 数据存储
        self.current_rgb = None
        self.current_depth = None
        self.current_result = None
        self.processing_steps = {}
        
        # 创建界面
        self.create_widgets()
        self.setup_layout()
        
        # 启动ROS2节点
        if ROS_AVAILABLE:
            self.start_ros_node()
        else:
            self.start_demo_mode()
            
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        self.main_frame = ttk.Frame(self.root)
        
        # 图像显示区域
        self.create_image_display()
        
        # 控制面板
        self.create_control_panel()
        
        # 参数调节面板
        self.create_parameter_panel()
        
        # 日志显示区域
        self.create_log_panel()
        
        # 状态栏
        self.create_status_bar()
        
    def create_image_display(self):
        """创建图像显示区域"""
        self.image_frame = ttk.LabelFrame(self.main_frame, text="图像显示", padding=10)
        
        # 创建图像显示标签
        self.rgb_label = ttk.Label(self.image_frame, text="RGB图像")
        self.rgb_canvas = tk.Canvas(self.image_frame, width=320, height=240, bg='black')
        
        self.depth_label = ttk.Label(self.image_frame, text="深度图像")
        self.depth_canvas = tk.Canvas(self.image_frame, width=320, height=240, bg='black')
        
        self.result_label = ttk.Label(self.image_frame, text="检测结果")
        self.result_canvas = tk.Canvas(self.image_frame, width=320, height=240, bg='black')
        
        # 图像切换按钮
        self.display_mode = tk.StringVar(value="rgb")
        self.rgb_radio = ttk.Radiobutton(self.image_frame, text="显示RGB", 
                                        variable=self.display_mode, value="rgb",
                                        command=self.update_display_mode)
        self.depth_radio = ttk.Radiobutton(self.image_frame, text="显示深度", 
                                          variable=self.display_mode, value="depth",
                                          command=self.update_display_mode)
        
    def create_control_panel(self):
        """创建控制面板"""
        self.control_frame = ttk.LabelFrame(self.main_frame, text="系统控制", padding=10)
        
        # 系统状态
        self.status_var = tk.StringVar(value="未连接")
        self.status_label = ttk.Label(self.control_frame, text="状态:")
        self.status_display = ttk.Label(self.control_frame, textvariable=self.status_var)
        
        # 控制按钮
        self.connect_btn = ttk.Button(self.control_frame, text="连接系统", 
                                     command=self.connect_system)
        self.start_btn = ttk.Button(self.control_frame, text="开始识别", 
                                   command=self.start_recognition, state='disabled')
        self.stop_btn = ttk.Button(self.control_frame, text="停止识别", 
                                  command=self.stop_recognition, state='disabled')
        self.exit_btn = ttk.Button(self.control_frame, text="退出系统", 
                                  command=self.exit_system, state='disabled')
        
        # 形状选择
        self.shape_frame = ttk.LabelFrame(self.control_frame, text="识别形状选择")
        self.sphere_var = tk.BooleanVar(value=True)
        self.cuboid_var = tk.BooleanVar(value=True)
        self.cylinder_var = tk.BooleanVar(value=True)
        
        self.sphere_check = ttk.Checkbutton(self.shape_frame, text="球体", 
                                           variable=self.sphere_var,
                                           command=self.update_shape_selection)
        self.cuboid_check = ttk.Checkbutton(self.shape_frame, text="长方体", 
                                           variable=self.cuboid_var,
                                           command=self.update_shape_selection)
        self.cylinder_check = ttk.Checkbutton(self.shape_frame, text="圆柱体", 
                                             variable=self.cylinder_var,
                                             command=self.update_shape_selection)
        
    def create_parameter_panel(self):
        """创建参数调节面板"""
        self.param_frame = ttk.LabelFrame(self.main_frame, text="参数调节", padding=10)
        
        # 检测参数
        ttk.Label(self.param_frame, text="最小轮廓面积:").grid(row=0, column=0, sticky='w')
        self.min_area_var = tk.IntVar(value=300)
        self.min_area_scale = ttk.Scale(self.param_frame, from_=100, to=1000, 
                                       variable=self.min_area_var, orient='horizontal')
        self.min_area_label = ttk.Label(self.param_frame, textvariable=self.min_area_var)
        
        ttk.Label(self.param_frame, text="深度标准差阈值:").grid(row=1, column=0, sticky='w')
        self.depth_std_var = tk.DoubleVar(value=53.0)
        self.depth_std_scale = ttk.Scale(self.param_frame, from_=20.0, to=100.0, 
                                        variable=self.depth_std_var, orient='horizontal')
        self.depth_std_label = ttk.Label(self.param_frame, textvariable=self.depth_std_var)
        
        # 标定参数显示
        self.calib_frame = ttk.LabelFrame(self.param_frame, text="标定参数")
        self.calib_text = scrolledtext.ScrolledText(self.calib_frame, width=30, height=8)
        
    def create_log_panel(self):
        """创建日志面板"""
        self.log_frame = ttk.LabelFrame(self.main_frame, text="系统日志", padding=10)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, width=50, height=10)
        
        # 日志控制
        self.log_control_frame = ttk.Frame(self.log_frame)
        self.clear_log_btn = ttk.Button(self.log_control_frame, text="清除日志", 
                                       command=self.clear_log)
        self.save_log_btn = ttk.Button(self.log_control_frame, text="保存日志", 
                                      command=self.save_log)
        
    def create_status_bar(self):
        """创建状态栏"""
        self.status_frame = ttk.Frame(self.root)
        self.fps_var = tk.StringVar(value="FPS: 0")
        self.fps_label = ttk.Label(self.status_frame, textvariable=self.fps_var)
        
        self.detection_count_var = tk.StringVar(value="检测到: 0 个物体")
        self.detection_label = ttk.Label(self.status_frame, textvariable=self.detection_count_var)
        
    def setup_layout(self):
        """设置布局"""
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 图像显示区域布局
        self.image_frame.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        self.rgb_label.grid(row=0, column=0, pady=5)
        self.rgb_canvas.grid(row=1, column=0, padx=5)
        
        self.depth_label.grid(row=0, column=1, pady=5)
        self.depth_canvas.grid(row=1, column=1, padx=5)
        
        self.result_label.grid(row=0, column=2, pady=5)
        self.result_canvas.grid(row=1, column=2, padx=5)
        
        self.rgb_radio.grid(row=2, column=0, pady=5)
        self.depth_radio.grid(row=2, column=1, pady=5)
        
        # 控制面板布局
        self.control_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        self.status_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.status_display.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        self.connect_btn.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        self.start_btn.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        self.stop_btn.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        self.exit_btn.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
        # 形状选择布局
        self.shape_frame.grid(row=5, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        self.sphere_check.grid(row=0, column=0, sticky='w')
        self.cuboid_check.grid(row=1, column=0, sticky='w')
        self.cylinder_check.grid(row=2, column=0, sticky='w')
        
        # 参数面板布局
        self.param_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        
        self.min_area_scale.grid(row=0, column=1, sticky='ew', padx=5)
        self.min_area_label.grid(row=0, column=2, padx=5)
        
        self.depth_std_scale.grid(row=1, column=1, sticky='ew', padx=5)
        self.depth_std_label.grid(row=1, column=2, padx=5)
        
        self.calib_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)
        self.calib_text.pack(fill='both', expand=True)
        
        # 日志面板布局
        self.log_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        self.log_text.pack(fill='both', expand=True)
        
        self.log_control_frame.pack(fill='x', pady=5)
        self.clear_log_btn.pack(side='left', padx=5)
        self.save_log_btn.pack(side='left', padx=5)
        
        # 状态栏布局
        self.status_frame.pack(side='bottom', fill='x')
        self.fps_label.pack(side='left', padx=10)
        self.detection_label.pack(side='left', padx=10)
        
        # 配置权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=2)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
    def start_ros_node(self):
        """启动ROS2节点"""
        if not ROS_AVAILABLE:
            self.log_message("ROS2不可用，无法启动节点")
            return
            
        try:
            rclpy.init()
            self.ros_node = ROS2ShapeRecognitionNode(self.ros_callback)
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.ros_node)
            
            # 在独立线程中运行ROS2
            self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.ros_thread.start()
            
            self.log_message("ROS2节点启动成功")
            self.status_var.set("已连接ROS2")
            
        except Exception as e:
            self.log_message(f"ROS2节点启动失败: {e}")
            self.status_var.set("ROS2连接失败")
    
    def start_demo_mode(self):
        """启动演示模式"""
        self.log_message("启动演示模式（无ROS2）")
        self.status_var.set("演示模式")
        
        # 创建演示数据
        self.create_demo_data()
        
    def create_demo_data(self):
        """创建演示数据"""
        # 生成示例RGB图像
        demo_rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(demo_rgb, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.putText(demo_rgb, "Demo Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 生成示例深度图像
        demo_depth = np.random.randint(100, 300, (240, 320), dtype=np.uint16)
        
        self.current_rgb = demo_rgb
        self.current_depth = demo_depth
        
        # 更新显示
        self.update_image_display()
        
    def ros_callback(self, event_type, data):
        """ROS2回调函数"""
        if event_type == 'camera_update':
            self.current_rgb = data['rgb']
            self.current_depth = data['depth']
            self.root.after(0, self.update_image_display)
            
        elif event_type == 'result_update':
            self.current_result = data['result']
            self.root.after(0, self.update_result_display)
            
    def update_image_display(self):
        """更新图像显示"""
        if self.current_rgb is not None:
            self.display_image_on_canvas(self.rgb_canvas, self.current_rgb)
            
        if self.current_depth is not None:
            # 将深度图转换为可视化图像
            depth_normalized = cv2.normalize(self.current_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            self.display_image_on_canvas(self.depth_canvas, depth_colored)
            
    def update_result_display(self):
        """更新结果显示"""
        if self.current_result is not None:
            self.display_image_on_canvas(self.result_canvas, self.current_result)
            
    def display_image_on_canvas(self, canvas, image):
        """在画布上显示图像"""
        try:
            # 调整图像尺寸
            height, width = image.shape[:2]
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # 计算缩放比例
                scale = min(canvas_width / width, canvas_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # 调整图像尺寸
                resized_image = cv2.resize(image, (new_width, new_height))
                
                # 转换为PIL图像
                if len(resized_image.shape) == 3:
                    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(resized_image)
                
                # 转换为PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # 更新画布
                canvas.delete("all")
                canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
                canvas.image = photo  # 保持引用
                
        except Exception as e:
            self.log_message(f"图像显示错误: {e}")
    
    def connect_system(self):
        """连接系统"""
        if not self.ros_node:
            self.log_message("ROS2节点未初始化")
            return
            
        try:
            # 调用enter服务
            request = Trigger.Request()
            response = self.ros_node.call_service(self.ros_node.enter_client, request)
            
            if response and response.success:
                self.log_message("系统连接成功")
                self.status_var.set("已连接")
                self.connect_btn['state'] = 'disabled'
                self.start_btn['state'] = 'normal'
                self.exit_btn['state'] = 'normal'
            else:
                self.log_message("系统连接失败")
                
        except Exception as e:
            self.log_message(f"连接系统时出错: {e}")
    
    def start_recognition(self):
        """开始识别"""
        if not self.ros_node:
            self.log_message("ROS2节点未初始化")
            return
            
        try:
            # 设置识别形状
            self.update_shape_selection()
            
            # 调用开始服务
            request = SetBool.Request()
            request.data = True
            response = self.ros_node.call_service(self.ros_node.set_running_client, request)
            
            if response and response.success:
                self.log_message("开始物体识别")
                self.status_var.set("识别中")
                self.start_btn['state'] = 'disabled'
                self.stop_btn['state'] = 'normal'
            else:
                self.log_message("启动识别失败")
                
        except Exception as e:
            self.log_message(f"启动识别时出错: {e}")
    
    def stop_recognition(self):
        """停止识别"""
        if not self.ros_node:
            return
            
        try:
            request = SetBool.Request()
            request.data = False
            response = self.ros_node.call_service(self.ros_node.set_running_client, request)
            
            if response:
                self.log_message("停止物体识别")
                self.status_var.set("已连接")
                self.start_btn['state'] = 'normal'
                self.stop_btn['state'] = 'disabled'
                
        except Exception as e:
            self.log_message(f"停止识别时出错: {e}")
    
    def exit_system(self):
        """退出系统"""
        if not self.ros_node:
            return
            
        try:
            request = Trigger.Request()
            response = self.ros_node.call_service(self.ros_node.exit_client, request)
            
            if response:
                self.log_message("退出系统")
                self.status_var.set("未连接")
                self.connect_btn['state'] = 'normal'
                self.start_btn['state'] = 'disabled'
                self.stop_btn['state'] = 'disabled'
                self.exit_btn['state'] = 'disabled'
                
        except Exception as e:
            self.log_message(f"退出系统时出错: {e}")
    
    def update_shape_selection(self):
        """更新形状选择"""
        if not self.ros_node:
            return
            
        try:
            shapes = []
            if self.sphere_var.get():
                shapes.append('sphere')
            if self.cuboid_var.get():
                shapes.append('cuboid')
            if self.cylinder_var.get():
                shapes.append('cylinder')
                
            request = SetStringList.Request()
            request.data = shapes
            response = self.ros_node.call_service(self.ros_node.set_shape_client, request)
            
            if response and response.success:
                self.log_message(f"更新识别形状: {', '.join(shapes)}")
            else:
                self.log_message("更新形状选择失败")
                
        except Exception as e:
            self.log_message(f"更新形状选择时出错: {e}")
    
    def update_display_mode(self):
        """更新显示模式"""
        if not self.ros_node:
            return
            
        try:
            mode = self.display_mode.get()
            request = SetBool.Request()
            request.data = (mode == "rgb")
            
            response = self.ros_node.call_service(self.ros_node.rgb_depth_client, request)
            if response:
                self.log_message(f"切换显示模式: {mode}")
                
        except Exception as e:
            self.log_message(f"切换显示模式时出错: {e}")
    
    def log_message(self, message):
        """记录日志消息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        print(log_entry.strip())  # 同时打印到控制台
    
    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """保存日志"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"日志已保存到: {filename}")
        except Exception as e:
            self.log_message(f"保存日志失败: {e}")
    
    def on_closing(self):
        """关闭程序时的清理"""
        try:
            if self.ros_node:
                self.exit_system()
                
            if self.executor:
                self.executor.shutdown()
                
            if ROS_AVAILABLE:
                rclpy.shutdown()
                
        except Exception as e:
            print(f"关闭时出错: {e}")
        finally:
            self.root.destroy()
    
    def run(self):
        """运行主程序"""
        self.log_message("形状识别上位机启动")
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = ShapeRecognitionGUI()
        app.run()
    except KeyboardInterrupt:
        print("程序被中断")
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()