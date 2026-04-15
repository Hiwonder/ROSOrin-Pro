#!/usr/bin/env python3
# encoding: utf-8
# @Author: Gcusms
# @Date: 2025/11/08
import os
import re
import cv2
import ast
import time
import json
import math
import rclpy
import queue
import textwrap
import threading
import numpy as np
import message_filters
from cv_bridge import CvBridge

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Float32
from std_srvs.srv import Trigger, SetBool, Empty

from interfaces.srv import SetPose2D
from interfaces.srv import SetString as SetColor

from speech import speech
from large_models.config import *
from large_models_msgs.msg import Tools
from large_models_msgs.srv import SetModel, SetString, SetTools

from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# 物体追踪(Object tracking)
import pycuda.driver as cuda
cuda.init()  # 确保CUDA已经初始化(Ensure CUDA is initialized)
from large_models_examples.track_anything import ObjectTracker
from large_models_examples.tracker import Tracker

# arm
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from servo_controller.bus_servo_control import set_servo_position

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_pixel_distance",
            "description": "获取一个或多个指定位置像素的深度距离(Get depth distance of one or more specified pixel positions)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixel_position": {
                        "type": "string",
                        "description": "包含一个或多个像素坐标列表的JSON字符串(JSON string containing one or more pixel coordinate lists)",
                    }
                },
                "required": ["pixel_position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_obejct_pixel",
            "description": "识别物体，并获取他的像素位置(Identify objects and get their pixel positions)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "用户的问题(User's question)",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_locations",
            "description": "查询并列出机器人可以导航前往的所有预定义地点的位置列表(Query and list all predefined locations the robot can navigate to)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "查询并获取机器人当前在地图中的精确位置坐标和朝向(Query and get the robot's current precise position and orientation on the map)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_current_view",
            "description": "详细描述机器人摄像头当前看到的画面内容，以回答用户提出的具体问题,一般用在移动到某个地点之后(Describe in detail what the robot camera currently sees to answer user's specific questions)",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "关于当前画面的具体问题(Specific questions about the current view)",
                    }
                },
                "required": ["question"]
            }
        }
    },
    {   "type": "function",
        "function": {
            "name": "move_to_location",
            "description": "将机器人移动到指定位置(Move the robot to specified location)",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "要去的目标地点名称(Target location name)",
                        "enum": ["书房(study_room)","卧室(bedroom)","水果超市(fruit_supermarket)","厨房(kitchen)","3号分拣台(sorting_station_3)","原点(Origin Point)"]
                    }
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "line_following",
            "description": "命令机器人沿着指定颜色的线进行巡线,或者寻检任务(Command robot to follow line of specified color)",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "要巡线的颜色(Color to follow)",
                        "enum": ["red", "green", "blue", "black", "yellow"]
                    }
                },
                "required": ["color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lidar_scan_detect",
            "description": "巡线过程当中，根据激光雷达扫描结果进行障碍物检测(Obstacle detection during line following based on LiDAR scan)",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_detect": {
                        "type": "string",
                        "description": "障碍物检测结果(Obstacle detection result)",
                    }
                },
                "required": ["scan_detect"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "object_track_detect",
            "description": "命令机器人追踪目标物体，注意：只有是确定追踪时才进行调用,而且不需要指定特定的颜色，前提是需要获取目标在图线上的像素方框位置(Object tracking command, note that only when the robot is sure to track the object, and no need to specify a specific color. Need to get the target object position in the image line pixel square.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "box": {
                        "type": "string",
                        "description": "返回目标在画面的方框定位，例如[xmin, ymin, xmax, ymax]",
                    }
                },
                "required": ["box"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "color_track",
            "description": "命令机器人追踪指定颜色的物体(Command robot to track object of specified color)",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "要追踪的物体的颜色(Color of object to track)",
                        "enum": ["red", "green", "blue", "black", "yellow"]
                    }
                },
                "required": ["color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "robot_move_control",
            "description": "以指定的线速度和角速度，控制机器人移动一段特定的时间,旋转90°所用时是4s(Move robot with specified linear speed and angular speed for a certain time, it takes 8s to rotate 90°)",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": "X轴方向的线速度,默认是0.1m/s(Linear velocity in X direction)",
                    },
                    "linear_y": {
                        "type": "number",
                        "description": "Y轴方向的线速度,默认是0.1m/s(Linear velocity in Y direction)",
                    },
                    "angular_z": {
                        "type": "number",
                        "description": "Z轴的角速度,默认是0.5m/s(Angular velocity in Z direction)",
                    },
                    "duration": {
                        "type": "number",
                        "description": "移动的持续时间(Movement duration)",
                    }
                },
                "required": ["linear_x", "linear_y", "angular_z", "duration"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_object_box",
            "description": "命令机器人追踪特定的目标物体(Command robot to track specific target object)",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj_track": {
                        "type": "string",
                        "description": "用户的问题(User's question)"
                    }
                },
                "required": ["obj_track"]
            }
        }
    },
]

position_dict = {
    "书房(study_room)": [1.2, -0.8, 0.0, 0.0, 80.0],
    "卧室(bedroom)": [1.38, 0.50, 0.0, 0.0, 10.0],
    "水果超市(fruit_supermarket)": [3.02, -0.49, 0.0, 0.0, -90.0],
    "厨房(kitchen)": [0.05, 0.0, 0.0, 0.0, 0.0],
    "3号分拣台(sorting_station_3)": [1.15, 0.1, 0.0, 0.0, 0.0],
    "原点(Origin Point)": [0.0, 0.0, 0.0, 0.0, 0.0],
}

if os.environ.get('ASR_LANGUAGE') == 'English':
    content_string = textwrap.dedent("""
        # Role setting
        You are a real interactive robot, need to execute tasks according to user instructions,
        and interact with users in a friendly way, just like chatting with friends.
        ## Workflow
        1. **Task Planning:** Before starting a task, you need to break it down and plan it. 
        The steps will be presented in a numbered format, with each number representing an independent step.
        2. **Tool Usage:** Before each tool is used, you need to provide an explanation, no more than 20 words, 
        describing the feedback in a humorous and varied way to make the communication process more engaging.
        3. **Feedback Processing:** After the tool is used, you need to follow up with a commentary on the feedback results, 
        no more than 20 words, describing the feedback in a humorous and varied way to make the communication process more engaging.
        4. **Task Completion:** After all task steps have been completed, provide a summary explanation, no more than 40 words.
        5. **Answer in english
    """) 

    PROMPT = '''
    As an image recognition expert, your capability is to accurately locate objects in images sent by users through object detection, and output the final results according to the "Output Format".
    ## 1. Understand User Instructions
    I will give you a sentence. You need to make the best decision based on my words and extract the "object name" from the decision. **The name corresponding to the object must be in English**, **do not output objects that are not mentioned**.
    ## 2. Understand the Image
    I will give you an image. Analyze the image and identify all recognizable objects within it.
    For each identified object, calculate the center point coordinates of the object. **Do not output objects that are not mentioned**.
    【Special Note】: Deeply understand the positional relationships of objects.
    ## Output Format (Please only output the following content, do not say any extra words)
    [
    {
    "object": name_1,
    "center_xy": [center_x_1, center_y_1]
    },
    {
    "object": name_2,
    "center_xy": [center_x_2, center_y_2]
    }
    ]
    '''

    OBJ_TRACK_PROMPT = '''
    As an intelligent vehicle, skilled in image recognition, your capability is to accurately locate objects in images sent by users through object detection, output the final results according to the "Output Format", and then perform tracking.
    ## 1. Understand User Instructions
    I will give you a sentence. You need to extract the "object name" from my words. **The name corresponding to the object must be in English**, **do not output objects that are not mentioned**.
    ## 2. Understand the Image
    I will give you an image. From this image, find the pixel coordinates of the top-left and bottom-right corners of the object corresponding to the "object name". If not found, then xyxy should be []. **Do not output objects that are not mentioned**.
    【Special Note】: Deeply understand the positional relationships of objects. The response needs to combine the user's instruction and the detection results.
    ## Output Format (Please only output the following content, do not say any extra words)
    {
        "object": "name", 
        "xyxy": [xmin, ymin, xmax, ymax]
    }
    '''
else:

    content_string = textwrap.dedent("""
        # 角色设定
        你是一个真实可交互的机器人, 需依据用户指令执行任务,并以第一人称与用户亲切交流,就像和朋友聊天一样。
        ## 工作流程
        1. **规划任务**：在进行任务前你需要先进行任务拆解和规划,步骤会以序号形式呈现,每个序号代表一个独立步骤。
        2. **调用工具**：在每个工具被调用前需要进行说明，不超过20字,说明风趣且变化无穷的反馈信息，让交流过程妙趣横生。
        3. **处理反馈**：工具调用完成后，需要跟进反馈的结果进行说明，不超过20字,说明风趣且变化无穷的反馈信息，让交流过程妙趣横生。
        4. **完成任务**：当全部任务步骤执行完成后进行总结说明，不超过40字。
    """)


    PROMPT = '''
    你作为图像识别专家，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出。
    ## 1. 理解用户指令
    我会给你一句话，你需要根据我的话做出最佳决策，从做出的决策中提取「物体名称」, **object对应的name要用英文表示**, **不要输出没有提及到的物体**
    ## 2. 理解图片
    我会给你一张图, 分析图片，找出其中所有可识别的物体。
    对于每一个被识别出的物体，并计算出物体的中心点坐标,**不要输出没有提及到的物体**
    【特别注意】： 要深刻理解物体的方位关系
    ## 输出格式（请仅输出以下内容，不要说任何多余的话)
    [
    {
    "object": name_1,
    "center_xy": [center_x_1, center_y_1]
    },
    {
    "object": name_2,
    "center_xy": [center_x_2, center_y_2]
    }
    ]
    '''

    OBJ_TRACK_PROMPT = '''
    你作为智能车，善于图像识别，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出，然后进行跟随。
    ## 1. 理解用户指令
    我会给你一句话，你需要根据我的话中提取「物体名称」。 **object对应的name要用英文表示**, **不要输出没有提及到的物体**
    ## 2. 理解图片
    我会给你一张图, 从这张图中找到「物体名称」对应物体的左上角和右下角的像素坐标; 如果没有找到，那xyxy为[]。**不要输出没有提及到的物体**
    【特别注意】： 要深刻理解物体的方位关系, response需要结合用户指令和检测的结果进行回答
    ## 输出格式（请仅输出以下内容，不要说任何多余的话)
    {
        "object": "name", 
        "xyxy": [xmin, ymin, xmax, ymax]
    }
    '''

class LogColors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'

class LLMControlMove(Node):
    def __init__(self, name):
        super().__init__(name)
        self.initialize_variables()       # 初始化变量(initialization of variables)
        self.setup_ros_components()       # 设置ROS组件(setup of ROS components)
        self.setup_services_and_clients() # 设置服务客户端(setup of services and clients)
        self.setup_subs_and_pubs()        # 设置订阅者和发布者(setup of subscribers and publishers)
        self.setup_timers()               # 设置计时器(setup of timers)

    def initialize_variables(self):
        """初始化所有类变量(Initialize all class variables)"""
        self.language = os.environ.get('ASR_LANGUAGE')
        self.machine_type = os.environ.get('MACHINE_TYPE')

        self.tools = []
        self.vllm_result = ''
        self.current_pose = None
        self.saved_pose = None
        self.obstacle_detected = False
        self.bridge = CvBridge()

        self.action = []
        self.llm_result = ''
        self.running = True
        self.interrupt = False
        self.action_finish = False
        self.play_audio_finish = False
        self.is_task_running = False
        self.reach_goal = False

        self.cb_group = ReentrantCallbackGroup()

        # 线程安全锁(lock)
        self.draw_lock = threading.Lock()
        self.draw_flag = False

        # 巡线相关变量(Line following related variables)
        self.line_following_count = 0
        self.line_following_start = True
        self.current_left = 0.0
        self.current_right = 0.0

        # 物体追踪相关变量(Object tracking related variables)
        self.start_track = False
        self.track_box_p1 = None
        self.track_box_p2 = None
        self.object_detect_box = False
        self.get_box_flag = False
        self.box = None
        self.set_above = False

        # 图像队列(Image queue)
        self.image_pair_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=2)

        if self.language == 'Chinese':
            self.vllm_model_name = 'qwen-vl-max-latest'
        elif self.language == 'English':
            self.vllm_model_name = vllm_model


    def setup_ros_components(self):
        """设置ROS2组件(Setup ROS2 components)"""
        # 物体追踪器(Object tracker)
        self.track = ObjectTracker(use_mouse=True, automatic=True, log=self.get_logger())
        
        # PID参数(PID parameters)
        self.pid_params = {
            'kp1': 0.05, 'ki1': 0.0, 'kd1': 0.00,
            'kp2': 0.002, 'ki2': 0.0, 'kd2': 0.0,
        }
        
        for param_name, default_value in self.pid_params.items():
            self.declare_parameter(param_name, default_value)
            self.pid_params[param_name] = self.get_parameter(param_name).value

        self.track.update_pid(
            [self.pid_params['kp1'], self.pid_params['ki1'], self.pid_params['kd1']],
            [self.pid_params['kp2'], self.pid_params['ki2'], self.pid_params['kd2']]
        )

    def setup_services_and_clients(self):
        """设置服务和服务客户端(Setup services and clients)"""
        # LLM相关客户端(LLM related clients)
        if self.language == 'English':
            self.client = speech.OpenAIAPI(vllm_api_key, vllm_base_url)
        else:
            self.client = speech.OpenAIAPI(api_key, base_url)
        self.set_tool_client = self.create_client(SetTools, 'agent_process/set_tool')
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        
        # 导航客户端(Navigation client)
        self.set_pose_client = self.create_client(SetPose2D, 'navigation_controller/set_pose')
        
        # 巡线客户端(Line following clients)
        self.line_follower_enter_client = self.create_client(Trigger, 'line_following/enter')
        self.line_follower_exit_client = self.create_client(Trigger, 'line_following/exit')
        self.line_follower_start_client = self.create_client(SetBool, 'line_following/set_running')
        self.line_follower_set_target_client = self.create_client(SetColor, 'line_following/set_color')
        
        # 颜色追踪客户端(Color tracking clients)
        self.object_tracker_enter_client = self.create_client(Trigger, 'object_tracking/enter')
        self.object_tracker_start_client = self.create_client(SetBool, 'object_tracking/set_running')
        self.object_tracker_set_target_client = self.create_client(SetColor, 'object_tracking/set_color')
        
        # 语音唤醒客户端(Voice wake-up client)
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')

    def setup_subs_and_pubs(self):
        """设置订阅者和发布者(Setup subscribers and publishers)"""
        # LLM相关(LLM related)
        self.create_subscription(Tools, 'agent_process/tools', self.tools_callback, 1, callback_group=self.cb_group)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.tools_result_pub = self.create_publisher(Tools, 'agent_process/tools_result', 1)

        # 导航相关(Navigation related)
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, qos_profile)
        self.create_subscription(Bool, 'navigation_controller/reach_goal', self.reach_goal_callback, 1, callback_group=self.cb_group)

        # 语音相关(Voice related)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1, callback_group=self.cb_group)
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=self.cb_group)

        # 底盘控制(Chassis control)
        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        # 巡线相关(Line following)
        self.cmd_vel_subscription = self.create_subscription(Twist,'/controller/cmd_vel',self.cmd_vel_callback,10)


        # 初始化姿态(Initial posture)
        if 'Pro' in self.machine_type:
            self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
            set_servo_position(self.joints_pub, 1, ((10, 500), (5, 500), (4, 200), (3, 50), (2, 750), (1, 500)))  # 初始姿态
        
        # 图像同步(Image synchronization)
        self.camera = os.environ.get('DEPTH_CAMERA_TYPE','depth_cam')
        if self.camera != 'usb_cam':
            depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth0/image_raw')
            image_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb0/image_raw')
            ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 3, 0.02)
            ts.registerCallback(self.image_sync_callback)
        else:
            self.create_subscription(Image, '/depth_cam/depth0/image_raw', self.image_callback, 1)

    def setup_timers(self):
        """设置定时器(Setup timers)"""
        self.timer = self.create_timer(0.0, self.init_process, callback_group=self.cb_group)

    def wait_for_services(self, timeout_sec=5.0):
        """等待关键服务就绪(Wait for key services to be ready)"""
        services = [
            (self.set_model_client, 'set_model'),
            (self.set_prompt_client, 'set_prompt'),
            (self.set_tool_client, 'set_tool'),
            (self.line_follower_enter_client, 'line_follower_enter'),
            (self.line_follower_set_target_client, 'line_follower_set_target'),
            (self.line_follower_start_client, 'line_follower_start'),
            (self.object_tracker_enter_client, 'object_tracker_enter'),
            (self.object_tracker_start_client, 'object_tracker_start'),
            (self.object_tracker_set_target_client, 'object_tracker_set_target')
        ]

        for client, name in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn(f'Service {name} not available after {timeout_sec} seconds')

    def amcl_pose_callback(self, msg):
        """处理AMCL位姿信息(Process AMCL pose information)"""
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        
        # 四元数转欧拉角(Quaternion to Euler angles)
        t3 = +2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        t4 = +1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw_z = math.atan2(t3, t4)
        yaw_deg = math.degrees(yaw_z)

        self.current_pose = {
            "x": position.x,
            "y": position.y,
            "yaw_degrees": yaw_deg
        }

    def reach_goal_callback(self, msg):
        """到达目标回调(Reach goal callback)"""
        self.get_logger().info('Reached goal')
        self.reach_goal = msg.data

    def llm_result_callback(self, msg):
        """LLM结果回调(LLM result callback)"""
        self.llm_result = msg.data
        self.get_logger().info(f'{LogColors.YELLOW}{LogColors.BOLD}LLM Reply: {self.llm_result}{LogColors.RESET}')

        # 非列表响应才进行语音播报(Only non-list responses are spoken)
        text_to_speak = self.llm_result
        is_list_response = re.search(r'^\s*\d+\.', text_to_speak, re.MULTILINE)
        if not is_list_response:
            tts_msg = String()
            tts_msg.data = text_to_speak
            self.tts_text_pub.publish(tts_msg)

    def image_sync_callback(self, ros_image, ros_depth_image):
        """同步图像回调(Synchronized image callback)"""
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            depth_image = np.ndarray(
                shape=(ros_depth_image.height, ros_depth_image.width), 
                dtype=np.uint16, 
                buffer=ros_depth_image.data
            )

            if self.image_pair_queue.full():
                self.image_pair_queue.get()
            self.image_pair_queue.put((bgr_image, depth_image))
        except Exception as e:
            self.get_logger().error(f"Image sync error: {str(e)}")

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # If the queue is full, discard the oldest image(如果队列已满，丢弃最旧的图像)
            self.image_queue.get()
        # Put the image into the queue(将图像放入队列)
        self.image_queue.put(bgr_image)

    def get_pixel_distance(self, pixel_coords_list):
        """获取像素距离(Get pixel distance)"""
        try:
            if self.camera != 'usb_cam':
                _, depth_image = self.image_pair_queue.get()
            else:
                return '请使用深度摄像头(Please use a depth camera)'
            distances = []
            
            for pixel_xy in pixel_coords_list:

                x, y = pixel_xy
                
                roi = [y, y + 5, x, x + 5]

                img_h, img_w = depth_image.shape[:2]
                if roi[0] < 0: roi[0] = 0
                if roi[1] > img_h: roi[1] = img_h
                if roi[2] < 0: roi[2] = 0
                if roi[3] > img_w: roi[3] = img_w
                
                roi_distance = depth_image[roi[0]:roi[1], roi[2]:roi[3]]
                
                valid_distances = roi_distance[np.logical_and(roi_distance > 0, roi_distance < 30000)]
                
                if len(valid_distances) > 0:
                    distance = round(float(np.mean(valid_distances) / 1000.0), 3)
                else:
                    distance = 0.0
                distances.append(distance)
            return str(distances)
        except Exception as e:
            self.get_logger().error(f"Get pixel distance error: {str(e)}")
            return "[]"

    def get_obejct_pixel(self, user_query):
        """获取物体像素位置(Get object pixel position)"""
        try:
            rgb_image, _ = self.image_pair_queue.get()
            vllm_result_str = self.client.vllm(user_query, rgb_image, prompt=PROMPT, model=self.vllm_model_name)

            # 提取JSON部分(Extract JSON part)
            if "```json" in vllm_result_str:
                json_part = vllm_result_str.split("```json")[1].split("```")[0]
            else:
                json_part = vllm_result_str

            detected_objects = json.loads(json_part.strip())
            return str(detected_objects)
        except Exception as e:
            self.get_logger().error(f"Get object pixel error: {str(e)}")
            return "[]"

    def move_to_location(self, destination):
        """移动到指定位置(Move to specified location)"""
        if destination not in position_dict:
            return f"移动失败：未知的目标地点 '{destination}'。(Move failed: Unknown destination '{destination}'.)"

        self.reach_goal = False
        msg = SetPose2D.Request()
        p = position_dict[destination]
        msg.data.x = float(p[0])
        msg.data.y = float(p[1])
        msg.data.roll = p[2]
        msg.data.pitch = p[3]
        msg.data.yaw = p[4]
        
        self.send_request(self.set_pose_client, msg)
        self.get_logger().info(f"Navigation goal '{destination}' sent. Waiting for arrival...")

        # 等待到达目标(Wait for reaching goal)
        while not self.reach_goal:
            time.sleep(0.1)

        if self.reach_goal:
            return f"已成功抵达{destination}(Arrived at {destination})"
        else:
            return f"移动超时，未能到达{destination}(Timeout, failed to reach {destination})"

    def get_current_location(self):
        """获取当前位置(Get current location)"""
        timeout_sec = 10.0
        start_time = time.time()

        while True:
            if self.current_pose is None :
                if time.time() - start_time < timeout_sec:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)
                break
        self.get_logger().info(f'Current location: {self.current_pose}')
        if self.current_pose:
            x = self.current_pose['x']
            y = self.current_pose['y']
            location_string = self.find_nearest_location(x, y, position_dict)
            return location_string
        else:
            return "抱歉，我现在还无法确定自己的位置信息。(Sorry, I can't find my location now.)"

    def find_nearest_location(self, current_x, current_y, position_dict):
        """查找最近的位置(Find nearest location)"""
        min_distance = float('inf')
        nearest_location_name = None

        for location_name, coords in position_dict.items():
            target_x, target_y = coords[0], coords[1]
            distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_location_name = location_name

        if nearest_location_name:
            if min_distance < 0.2:
                return f'我现在在{nearest_location_name}(I am at {nearest_location_name})'
            else:
                return f'我现在在{nearest_location_name}附近(I am near {nearest_location_name})'
        else:
            return '我现在暂时不知道在哪里(I do not know where I am now)'

    def describe_current_view(self, question):
        """描述当前视图(Describe current view)"""
        try:
            if self.camera != 'usb_cam':
                rgb_image, _ = self.image_pair_queue.get(block=True)
            else:
                rgb_image = self.image_queue.get(block=True)
            if self.language == 'Chinese':
                VLLM_PROMPT = textwrap.dedent(f"""
                作为我的机器人管家，请仔细观察摄像头捕捉到的画面，并根据以下问题给出一个简洁、人性化的回答。
                不要进行反问，字数在10到40字之间。
                问题是："{question}
            """)
            else:
                VLLM_PROMPT = textwrap.dedent(f"""
                To be my robot butler, please observe the image captured by camera carefully. 
                Please give a concise and humanized answer to the following question. 
                Don't ask any questions back. The length of your answer should be between 10 to 40 words.
                "{question}
            """)
            description = self.client.vllm(question, rgb_image, prompt=VLLM_PROMPT, model=self.vllm_model_name)

            self.get_logger().info(f'{LogColors.YELLOW}{LogColors.BOLD}LLM Reply: {description}{LogColors.RESET}')

            # 直接发布到TTS进行语音播报
            tts_msg = String()
            tts_msg.data = description
            self.tts_text_pub.publish(tts_msg)

            return f"画面描述任务已成功执行。得到的结果是{description}(The description of the view has been successfully executed. The result is {description})"

        except Exception as e:
            self.get_logger().error(f"Describe current view error: {str(e)}")
            return "无法描述当前画面(Unable to describe the current view)"

    def line_following(self, color):
        """巡线功能(Line following function)"""
        self.line_follower_enter_client.call_async(Trigger.Request())
        self.line_following_start = True

        # 设置目标颜色(Set target color)
        color_msg = SetColor.Request()
        color_msg.data = color
        self.line_follower_set_target_client.call_async(color_msg)

        # 启动巡线(Start line following)
        start_msg = SetBool.Request()
        start_msg.data = True
        self.line_follower_start_client.call_async(start_msg)

        self.is_task_running = True
        return f"好的，马上开始沿着{color}线行驶。(Okay, I will follow the {color} line.)"

    def lidar_scan_detect(self, scan_detect):
        """激光雷达障碍物检测(LiDAR obstacle detection)"""
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}start lidar_scan_detect{LogColors.RESET}')
        time.sleep(2)

        stop_distance = 0.3
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        while self.line_following_start:
            if abs(self.current_linear_x) < 0.001 and abs(self.current_angular_z) < 0.001:
                if self.line_following_count < 500:
                    self.line_following_count += 1
                    if self.line_following_count % 20 == 0:
                        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Detected zero velocity{LogColors.RESET}')
                else:
                    start_msg = SetBool.Request()
                    start_msg.data = False
                    self.line_follower_start_client.call_async(start_msg) 
                    self.line_following_count = 0
                    time.sleep(2)
                    self.line_follower_exit_client.call_async(Trigger.Request())
                    time.sleep(1)
                    return '检测到速度为0,已停止巡线(Detected zero velocity, stop line following)'
            else:
                self.line_following_count = 0

            time.sleep(0.01)

        if not self.line_following_start:
            return '已停止巡线(Stop line following)'
        else:
            return '巡线检测超时(Line following timeout)'


    def cmd_vel_callback(self, msg):
        """cmd_vel话题回调函数(cmd_vel topic callback function)"""
        if self.line_following_start:
            self.current_linear_x = msg.linear.x
            self.current_angular_z = msg.angular.z

    def color_track(self, color):
        """颜色追踪(Color tracking)"""
        self.object_tracker_enter_client.call_async(Trigger.Request())

        # 设置要追踪的颜色(Set color to track)
        color_msg = SetColor.Request()
        color_msg.data = color
        self.object_tracker_set_target_client.call_async(color_msg)

        # 启动追踪(Start tracking)
        start_msg = SetBool.Request()
        start_msg.data = True
        self.object_tracker_start_client.call_async(start_msg)
        
        self.is_task_running = True
        return f"好的,马上开始追踪{color}的物体。(Okay, I'll start tracking {color} objects.)"

    def robot_move_control(self, linear_x, linear_y, angular_z, duration):
        """机器人移动控制(Robot movement control)"""
        self.get_logger().info(f"Executing move: x={linear_x}, y={linear_y}, z={angular_z} for {duration}s")
        self.is_task_running = True

        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.linear.y = float(linear_y)
        twist_msg.angular.z = float(angular_z)

        self.cmd_vel_pub.publish(twist_msg)

        end_time = time.time() + float(duration)
        while time.time() < end_time and self.is_task_running:
            time.sleep(0.05)

        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info("Movement finished, stopping robot.")
        return f"move: x={linear_x}, y={linear_y}, z={angular_z}, duration {duration}s"

    def get_object_box(self, user_query):
        """获取物体边界框(Get object bounding box)"""
        self.get_logger().info(f"ObjTracking: {user_query}")

        try:
            if self.camera != 'usb_cam':
                rgb_image, _ = self.image_pair_queue.get(block=True)
            else:
                rgb_image = self.image_queue.get(block=True)
            vllm_result_str = self.client.vllm(user_query, rgb_image, prompt=OBJ_TRACK_PROMPT, model=self.vllm_model_name)

            if "```json" in vllm_result_str:
                json_part = vllm_result_str.split("```json")[1].split("```")[0]
            else:
                json_part = vllm_result_str

            detected_objects = json.loads(json_part.strip())
            self.get_box_flag = True
            
            if 'xyxy' in detected_objects:
                self.box = detected_objects['xyxy']
                self.get_logger().info('Detected objects: %s' % str(self.box))
                self.is_task_running = True
            return f'find the object is' + str(detected_objects)
        except Exception as e:
            self.get_logger().error(f"Get object box error: {str(e)}")
            return "{}"


    def object_track_detect(self, box):
        """物体追踪(Object Track)"""
        self.get_logger().info('Object Tracking: %s' % str(box))
        with self.draw_lock:
            self.draw_flag = True
            self.box = ast.literal_eval(box)
        self.object_detect_box = True
        return 'Start Tracking'


    def get_available_locations(self):
        """获取可用位置(Get available locations)"""
        self.get_logger().info("Querying available locations.")
        return json.dumps(position_dict, ensure_ascii=False, indent=2)

    def get_node_state(self, request, response):
        """获取节点状态(Get node state)"""
        return response

    def init_process(self):
        """初始化过程(Initialization process)"""
        self.timer.cancel()
        self.wait_for_services()

        # 设置模型(Set model)
        msg = SetModel.Request()
        msg.model_type = 'llm_tools'
        if self.language == 'Chinese':
            msg.model = 'qwen3-max'
            msg.api_key = api_key 
            msg.base_url = base_url
        elif self.language == 'English':
            msg.model =  'qwen/qwen3-max'
            msg.api_key = vllm_api_key 
            msg.base_url = vllm_base_url
        self.send_request(self.set_model_client, msg)

        # 设置提示词(Set prompt)
        msg = SetString.Request()
        msg.data = content_string
        self.send_request(self.set_prompt_client, msg)

        # 设置工具(Set tools)
        tools_json = [json.dumps(tool, ensure_ascii=False) for tool in tools]
        msg = SetTools.Request()
        msg.tools = tools_json
        self.send_request(self.set_tool_client, msg)


        # 启动处理线程(Start processin threads)
        threading.Thread(target=self.process, daemon=True).start()
        threading.Thread(target=self.display_thread, daemon=True).start()
        threading.Thread(target=self.object_track_thread, daemon=True).start()

        # 启动功能开启语音提示(Start function opening voice prompt)
        speech.play_audio(start_audio_path)

        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def send_request(self, client, msg):
        """发送请求并等待响应(Send request and wait for response)"""
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done():
                try:
                    return future.result()
                except Exception as e:
                    self.get_logger().error(f"Service call failed: {str(e)}")
                    return None
            time.sleep(0.01)
        return None

    def tools_callback(self, msg):
        """工具回调(Tools callback)"""
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}AI Decision-Making:{LogColors.RESET}')
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Tools id [{msg.id}]:{LogColors.RESET}')
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Tools name [{msg.name}]:{LogColors.RESET}')
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}Tools data [{msg.data}]:{LogColors.RESET}')
        
        self.tools = [msg.id, msg.name, json.loads(msg.data)]

    def play_audio_finish_callback(self, msg):
        """音频播放完成回调(Audio playback finished callback)"""
        if msg.data:
            self.play_audio_finish = True
            awake_msg = SetBool.Request()
            awake_msg.data = True
            self.send_request(self.awake_client, awake_msg)

    def wakeup_callback(self, msg):
        """唤醒回调(Wakeup callback)"""
        if msg.data and self.is_task_running:
            self.get_logger().info('唤醒中断(Wakeup interrupt)')
            
            # 停止巡线(Stop line following)
            if self.line_following_start:
                request = SetBool.Request()
                request.data = False
                self.line_follower_start_client.call_async(request)
                self.line_following_start = False

            # 停止颜色追踪(Stop color tracking)
            if self.start_track:
                request = SetBool.Request()
                request.data = False
                self.object_tracker_start_client.call_async(request)
                self.start_track = False

            # 停止运动(Stop movement)
            self.cmd_vel_pub.publish(Twist())
            self.track.stop()
            self.is_task_running = False
            
        elif msg.data:
            self.get_logger().info('Wakeup received, but no interruptible task is running.')

    def process(self):
        """主处理循环(Main processing loop)"""
        while rclpy.ok():
            if self.tools:
                tool_id, tool_name, args_dict = self.tools
                res = None
                
                try:
                    if tool_name == 'get_obejct_pixel':
                        content = args_dict.get('content') 
                        if content:
                            res = self.get_obejct_pixel(content)
                    elif tool_name == 'get_pixel_distance':
                        position_str = args_dict.get('pixel_position')
                        if position_str:
                            positions_list = ast.literal_eval(position_str)
                            if (isinstance(positions_list, list) and 
                                all(isinstance(p, list) and len(p) == 2 for p in positions_list)):
                                res = self.get_pixel_distance(positions_list)
                    elif tool_name == 'move_to_location':
                        destination = args_dict.get('destination')
                        if destination:
                            res = self.move_to_location(destination)
                    elif tool_name == 'describe_current_view':
                        question = args_dict.get('question')
                        if question:
                            res = self.describe_current_view(question)
                    elif tool_name == 'get_available_locations':
                        res = self.get_available_locations()
                    elif tool_name == 'get_current_location':
                        res = self.get_current_location()
                    elif tool_name == 'line_following':
                        color = args_dict.get('color')
                        if color:
                            res = self.line_following(color)
                    elif tool_name == 'color_track':
                        color = args_dict.get('color')
                        if color:
                            res = self.color_track(color)
                    elif tool_name == 'object_track_detect':
                        box = args_dict.get('box')
                        if box:
                            res = self.object_track_detect(box)
                    elif tool_name == 'lidar_scan_detect':
                        scan_detect = args_dict.get('scan_detect')
                        if scan_detect:
                            res = self.lidar_scan_detect(scan_detect)
                    elif tool_name == 'robot_move_control':
                        if all(k in args_dict for k in ["linear_x", "linear_y", "angular_z", "duration"]):
                            res = self.robot_move_control(
                                linear_x=args_dict['linear_x'],
                                linear_y=args_dict['linear_y'],
                                angular_z=args_dict['angular_z'],
                                duration=args_dict['duration']
                            )
                    elif tool_name == 'get_object_box':
                        obj_track = args_dict.get('obj_track')
                        if obj_track:
                            res = self.get_object_box(obj_track)
                            
                    if res is not None:
                        self.tools_result_pub.publish(Tools(id=tool_id, name=tool_name, data=res))
                        
                except Exception as e:
                    self.get_logger().error(f"Tool {tool_name} execution error: {str(e)}")
                    res = f"工具执行错误: {str(e)}(The tool execution error: {str(e)})"
                    self.tools_result_pub.publish(Tools(id=tool_id, name=tool_name, data=res))
                
                self.tools = []
                time.sleep(2)
            else:
                time.sleep(0.02)


    def display_thread(self):
        """显示线程(show thread)"""
        while self.running:
            try:
                if self.camera != 'usb_cam':
                    rgb_image, depth_image = self.image_pair_queue.get(block=True, timeout=1)
                else:
                    rgb_image = self.image_queue.get(block=True, timeout=1)
                result_image = rgb_image.copy()

                with self.draw_lock:
                    if self.draw_flag:
                        # 物体追踪绘制(object tracking drawing)
                        if self.start_track and self.track_box_p1 is not None and self.track_box_p2 is not None:
                            cv2.rectangle(result_image, self.track_box_p1, self.track_box_p2, (0, 255, 0), 2)
                if self.camera != 'usb_cam':
                    sim_depth_image = np.clip(depth_image, 0, 2000).astype(np.float64)
                    sim_depth_image = sim_depth_image / 2000.0 * 255.0
                    
                    depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                    result_image = np.concatenate([result_image, depth_color_map, ], axis=1)
    
                cv2.imshow("result_image", result_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    self.running = False

            except queue.Empty:
                if not self.running:
                    break
                continue
            except Exception as e:
                self.get_logger().error(f"Display thread error: {str(e)}")
                continue


    def object_track_thread(self):
        """物体追踪线程(Object tracking thread)"""
        # 在当前线程创建一个新的 CUDA 上下文(Create new CUDA context in current thread)
        dev = cuda.Device(0)
        ctx = dev.make_context()
        try:
            model_path = os.path.split(os.path.realpath(__file__))[0]

            back_exam_engine_path = os.path.join(model_path, "../resources/models/nanotrack_backbone_exam.engine")
            back_temp_engine_path = os.path.join(model_path, "../resources/models/nanotrack_backbone_temp.engine")
            head_engine_path = os.path.join(model_path, "../resources/models/nanotrack_head.engine")
            tracker = Tracker(back_exam_engine_path, back_temp_engine_path, head_engine_path)

            while self.running:
                try:
                    if self.camera != 'usb_cam':
                        image, depth_image = self.image_pair_queue.get(block=True)
                    else:
                        image = self.image_queue.get(block=True)
                    img_h, img_w, _ = image.shape
                    if self.box is not None and len(self.box) > 0 and self.object_detect_box:
                        # 转换为 [x, y, width, height] 格式(Convert to [x, y, width, height] format)
                        box_wh = [self.box[0], self.box[1], self.box[2] - self.box[0], self.box[3] - self.box[1]]
                        self.track.set_track_target(tracker, box_wh, image)
                        self.start_track = True
                        self.box = []

                    if self.start_track and self.is_task_running:
                        self.data = self.track.track(tracker, image, depth_image)
                        image = self.data[-1]
                        with self.draw_lock:
                            self.track_box_p1 = self.data[2]
                            self.track_box_p2 = self.data[3]
                        
                        self.track_box_p1 = (max(0, self.track_box_p1[0]), min(img_w-1, self.track_box_p1[1]))
                        self.track_box_p2 = (max(0, self.track_box_p2[0]), min(img_h-1, self.track_box_p2[1]))
                        if isinstance(self.data[0], (int, float)) and isinstance(self.data[1], (int, float)):
                            twist = Twist()
                            twist.linear.x, twist.angular.z = float(self.data[0]), float(self.data[1])
                            if 'Acker' in self.machine_type:
                                if twist.angular.z < -math.radians(40):
                                    twist.angular.z = -math.radians(40)
                                elif twist.angular.z > math.radians(40):
                                    twist.angular.z = math.radians(40)
                                steering_angle = twist.angular.z
                                if steering_angle != 0:
                                    R = 0.145 / math.tan(steering_angle)
                                    twist.angular.z = twist.linear.x / R
                            self.cmd_vel_pub.publish(twist)
                        else:
                            self.get_logger().warn(f"Invalid track data: {self.data[0]}, {self.data[1]}")
                        
                except Exception as e:
                    self.get_logger().error(f"Object track thread error: {str(e)}")
                    continue

            cv2.destroyAllWindows()
        finally:
            # 确保上下文被正确释放(Ensure context is properly released)
            ctx.pop()

def main():
    rclpy.init()
    node = LLMControlMove('llm_control')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()
 
if __name__ == "__main__":
    main()
