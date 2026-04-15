#!/usr/bin/env python3
# encoding: utf-8
# @Author: Gcusms
# @Date: 2025/10/21
import os
import re
import cv2
import textwrap
import ast
import time
import json
import math
import yaml
import rclpy
import threading
from speech import speech

from rclpy.node import Node

import numpy as np
import queue
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from geometry_msgs.msg import Twist

from std_msgs.msg import String, Bool, Float32,Int32
from std_srvs.srv import Trigger, SetBool, Empty
from large_models_msgs.msg import Tools
from large_models.config import *
from large_models_msgs.srv import SetModel, SetString, SetTools,SetBox,SetContent

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from interfaces.srv import SetPose2D,SetPoint
from interfaces.srv import SetString as SetColor
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from servo_controller.action_group_controller import ActionGroupController


from sdk import common
from sdk.pid import PID


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_available_locations",
            "description": "查询并列出机器人可以导航前往的所有预定义地点的位置列表。(Query and list the location of all pre-defined places that the robot can navigate to.)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {"type": "function",
        "function": {
            "name": "road_network_navigation",
            "description": "通过路网节点导航到特定功能区域，如仓库或工地等(Navigate to a specific functional area such as warehouse or construction site via road network node.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "要去的路网地点名称(Name of the road network location to go)",
                        "enum": ["配送中心(Distribution Center)", "公园(Park)", "仓库(Warehouse)", "社区(Community)","商店(Store)"]
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_current_view",
            "description": "详细描述机器人当前看到的画面内容，以回答用户提出的具体问题。(Describe the detailed contents of the scene that the robot sees currently, to answer the specific question that the user asks.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "关于当前画面的具体问题，例如'前面的大门有没有关'？(A specific question about the current scene, for example 'Is the door closed?', or 'What is there?')'"
                    }
                },
                "required": ["question"]
            }
        }
    },

]


waypoint_dict = {
    "配送中心(Distribution Center)": 0,
    "公园(Park)": 2,
    "仓库(Warehouse)": 12,  #12  6
    "社区(Community)": 13,  #13  5
    "商店(Store)": 15,
}

language = os.environ.get('ASR_LANGUAGE','')
if language == 'Chinese':

    content_string = textwrap.dedent("""
        # 角色设定
        你是一个风趣幽默的机器人助手，用第一人称与用户亲切交流，就像和朋友聊天一样自然。

        # 核心规则（必须遵守）
        1. 在执行任何工具调用前，必须先给我一段简短风趣的提示文字
        2. 工具调用完成后，必须给我一段简短风趣的结果说明
        3. 所有给我的文字回复都要保持轻松有趣的风格
        4. 如果任务没有需要执行工具，同样的也要用一小段文字返回

        # 工作流程
            1. **任务规划**：先简要说明你的行动计划（20-30字），风格要风趣幽默
            2. **调用工具**：每次调用工具前必须给我提示（15-25字），说明你要做什么
            3. **处理反馈**：工具执行后必须给我结果说明（15-25字），分享进展或趣事
            4. **任务总结**：完成后进行风趣总结（30-50字）

        # 重要提醒
        - 每次工具调用前后都必须给我文字回复
        - 保持对话的连贯性和趣味性
    """)
else:
    content_string = textwrap.dedent("""
    # Role Setting
    You are a witty and humorous robot assistant, communicating with users in the first person as naturally as chatting with a friend.

    # Core Rules (Must Follow)
    1. Before executing any tool call, you must first give me a short, witty prompt.
    2. After the tool call is completed, you must give me a short, witty result description.
    3. All text replies to me should maintain a lighthearted and fun style.
    4. If a task does not require executing a tool, also return a short piece of text.
    5. Answer in English

    # CRITICAL RULE:
    If you decide to call any tool, you MUST first output a normal text message
    explaining your plan. Tool calls without prior text are strictly forbidden.

    # Workflow
        1. **Task Planning**: Briefly explain your action plan (10-15 words) in a witty and humorous style.
        2. **Calling Tools**: Before each tool call, you must give me a prompt (10-20 words) explaining what you are about to do.
        3. **Processing Feedback**: After the tool executes, you must give me a result description (10-15 words), sharing progress or an interesting tidbit.
        4. **Task Summary**: Provide a witty summary upon completion (10-15 words).

    # Important Reminders
    - You must give me a text reply before and after every tool call.
    - Maintain the coherence and fun of the conversation.
""")

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

        self.initialize_variables() # 初始化变量(init variables)
        self.setup_ros_components() # 设置ROS节点(init node)
        self.setup_services_and_clients() # 设置ROS服务客户端(init services and clients)
        self.setup_subs_and_pubs() # 设置ROS订阅者与发布者(init subs and pubs)
        self.setup_timers() # 设置ROS定时器(init timers)

    def initialize_variables(self):
        """初始化所有类变量(init all class variables)"""
        self.language = os.environ.get('ASR_LANGUAGE')
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
        self.task_finished = False

        self.machine_type = os.environ.get('MACHINE_TYPE', '')

        self.cb_group = ReentrantCallbackGroup()

        # 图像队列(queue)
        self.image_pair_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=2)
        self.bridge_box = CvBridge()

        self.camera = os.environ.get('DEPTH_CAMERA_TYPE','depth_cam')


        if self.language == 'Chinese':
            self.vllm_model_name = 'qwen-vl-max-latest'
        elif self.language == 'English':
            self.vllm_model_name = vllm_model

    def setup_ros_components(self):
        """设置ROS2组件(Setup ROS2 components)"""
        pass

    def setup_services_and_clients(self):
        """设置服务和服务客户端(Set Services and Clients)"""
        # LLM相关客户端(LLM-related clients)
        if self.language == 'English':
            self.client = speech.OpenAIAPI(vllm_api_key, vllm_base_url)
        else:
            self.client = speech.OpenAIAPI(api_key, base_url)
        self.set_tool_client = self.create_client(SetTools, 'agent_process/set_tool')
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')

        # 导航客户端(Navigation clients)
        self.set_pose_client = self.create_client(SetPose2D, 'navigation_controller/set_pose')
        
        # 语音唤醒客户端(Voice wakeup clients)
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

        self.create_subscription(Bool, '/task_finish', self.task_finish_callback, 1, callback_group=self.cb_group)
        self.create_subscription(Bool, '/road_network_navigator/reach_final', self.reach_goal_callback, 1, callback_group=self.cb_group)
        self.waypoint_pub = self.create_publisher(Int32, '/request_waypoint', 1)

        # 语音相关(Voice related)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1, callback_group=self.cb_group)
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=self.cb_group)

        # 底盘控制(Chassis control)
        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        # self.controller = ActionGroupController(
        #     self.joints_pub, 
        #     '/home/ubuntu/software/arm_pc/ActionGroups'
        # )

        # 图像发布(Image publisher)
        self.result_image_publisher = self.create_publisher(Image, '~/result_image', 10)

        # # 图像同步(Image synchronizer)
        if self.camera != 'usb_cam':
            depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth0/image_raw')
            image_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb0/image_raw')
            ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 3, 0.02)
            ts.registerCallback(self.image_sync_callback)
        else:
            self.create_subscription(Image, '/depth_cam/rgb0/image_raw', self.image_callback, 1)

    def setup_timers(self):
        """设置定时器(Timer)"""
        self.timer = self.create_timer(0.0, self.init_process, callback_group=self.cb_group)

    def _wait_for_services(self, timeout_sec=5.0):
        """等待关键服务就绪(Wait for critical services to be available)"""
        services = [
            (self.set_model_client, 'set_model'),
            (self.set_prompt_client, 'set_prompt'),
            (self.set_tool_client, 'set_tool'),
        ]
        
        for client, name in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn(f'Service {name} not available after {timeout_sec} seconds')

    def amcl_pose_callback(self, msg):
        """处理AMCL位姿信息(Slove the pose information from AMCL)"""
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation

        # 四元数转欧拉角(Convert quaternion to euler angle)
        t3 = +2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        t4 = +1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw_z = math.atan2(t3, t4)
        yaw_deg = math.degrees(yaw_z)

        self.current_pose = {
            "x": position.x,
            "y": position.y,
            "yaw_degrees": yaw_deg
        }

    def task_finish_callback(self, msg):
        if msg.data:
            self.task_finished = True

    def reach_goal_callback(self, msg):
        """到达目标回调(Arrived at goal)"""
        self.get_logger().info('Reached goal')
        self.reach_goal = msg.data
        self.get_logger().info(f'{LogColors.GREEN}{LogColors.BOLD}>>>Reached goal<<<: {self.reach_goal}{LogColors.RESET}')


    def llm_result_callback(self, msg):
        """LLM结果回调(LLM result)"""
        self.llm_result = msg.data
        self.get_logger().info(f'{LogColors.YELLOW}{LogColors.BOLD}LLM Reply: {self.llm_result}{LogColors.RESET}')

        # 非列表响应才进行语音播报(Speak out the response if it is not a list response)
        text_to_speak = self.llm_result
        is_list_response = re.search(r'^\s*\d+\.', text_to_speak, re.MULTILINE)
        if not is_list_response:
            tts_msg = String()
            tts_msg.data = text_to_speak
            self.tts_text_pub.publish(tts_msg)


    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # If the queue is full, discard the oldest image(如果队列已满，丢弃最旧的图像)
            self.image_queue.get()
        # Put the image into the queue(将图像放入队列)
        self.image_queue.put(bgr_image)

    def road_network_navigation(self, location):
        """路网节点导航(Road network node navigation)"""
        if location not in waypoint_dict:
            return f"导航失败：未知的路网地点 '{location}'。"
        
        node_id = waypoint_dict[location]
        self.reach_goal = False
        self.task_finished = False
        
        msg = Int32()
        msg.data = node_id
        self.waypoint_pub.publish(msg)
        
        self.get_logger().info(f"Road network goal '{location}' (ID: {node_id}) sent. Waiting for arrival...")
        
        # 假设底层节点到达后同样会发布 reach_goal 信号(Publish reach_goal signal)
        while not self.reach_goal:
            time.sleep(0.1)
        
        return f"已通过路网成功抵达{location}"


    def get_current_location(self):
        """获取当前位置(get current location)"""
        timeout_sec = 10.0
        start_time = time.time()

        while self.current_pose is None and (time.time() - start_time) < timeout_sec:
            time.sleep(0.1)

        if self.current_pose:
            x = self.current_pose['x']
            y = self.current_pose['y']
            location_string = self.find_nearest_location(x, y, waypoint_dict)
            return location_string
        else:
            return "抱歉，我现在还无法确定自己的位置信息。(Sorry I can't determine my location now.)"

    def find_nearest_location(self, current_x, current_y, waypoint_dict):
        """查找最近的位置(Find the nearest location)"""
        min_distance = float('inf')
        nearest_location_name = None

        for location_name, coords in waypoint_dict.items():
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
            return '我现在暂时不知道在哪里(I am not sure where I am now)'

    def describe_current_view(self, question):
        """描述当前视图(Describe the current view)"""
        try:
            time.sleep(2)
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
                Don't ask any questions back. The length of your answer should be beteen 10 to 40 words.
                "{question}
            """)
            description = self.client.vllm(question, rgb_image, prompt=VLLM_PROMPT, model=self.vllm_model_name)


            self.get_logger().info(f'{LogColors.YELLOW}{LogColors.BOLD}LLM Reply: {description}{LogColors.RESET}')

            # 直接发布到TTS进行语音播报(publish to tts)
            tts_msg = String()
            tts_msg.data = description
            self.tts_text_pub.publish(tts_msg)

            return f"画面描述任务已成功执行。得到的结果是{description}(The description of the view has been successfully executed. The result is {description})"

        except Exception as e:
            self.get_logger().error(f"Describe current view error: {str(e)}")
            return "无法描述当前画面(Unable to describe the current view)"


    def get_available_locations(self):
        """获取可用位置(get available locations)"""
        self.get_logger().info("Querying available locations.")
        return str(waypoint_dict)

    def get_node_state(self, request, response):
        """获取节点状态(Get node state)"""
        return response

    def init_process(self):
        """初始化过程(Initialization process)"""
        self.timer.cancel()
        self._wait_for_services()

        # 设置模型
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
        
        # self.controller.run_action('init')
        # time.sleep(1.5)

        # 启动处理线程(Start processing thread)
        threading.Thread(target=self.process, daemon=True).start()
        # threading.Thread(target=self.display_thread, daemon=True).start()

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
        if msg.data:
            self.get_logger().info('唤醒中断(Wakeup interrupt)')

        elif msg.data:
            self.get_logger().info('Wakeup received, but no interruptible task is running.')

    def process(self):
        """主处理循环(Main processing loop)"""
        while rclpy.ok():
            if self.tools:
                tool_id, tool_name, args_dict = self.tools
                res = None
                try:
                    if tool_name == 'describe_current_view':
                        question = args_dict.get('question')
                        if question:
                            res = self.describe_current_view(question)
                    elif tool_name == 'road_network_navigation':
                        location = args_dict.get('location')
                        if location:
                            res = self.road_network_navigation(location)
                    elif tool_name == 'get_available_locations':
                        res = self.get_available_locations()
                    elif tool_name == 'get_current_location':
                        res = self.get_current_location()

                    if res is not None:
                        self.tools_result_pub.publish(Tools(id=tool_id, name=tool_name, data=res))

                except Exception as e:
                    self.get_logger().error(f"Tool {tool_name} execution error: {str(e)}")
                    res = f"工具执行错误: {str(e)}(tool used failed{tool_name})"
                    self.tools_result_pub.publish(Tools(id=tool_id, name=tool_name, data=res))
                
                self.tools = []
                time.sleep(2)
            else:
                time.sleep(0.02)
    def display_thread(self):
        """显示线程(show thread)"""
        while self.running:
            try:
                try:
                    if self.camera != 'usb_cam':
                        rgb_image, depth_image = self.image_pair_queue.get(block=True, timeout=1)
                    else:
                        rgb_image = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                result_image = rgb_image.copy()
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
            # 如果队列已满，丢弃最旧的图像 (If the queue is full, discard the oldest image)
            self.image_queue.get()
        # 将图像放入队列(Put the image into the queue)
        self.image_queue.put(bgr_image)

def main():
    rclpy.init()
    node = LLMControlMove('road_network_tool')
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
