#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2025/03/06
import os
import re
import time
import rclpy
import threading
from speech import speech
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty

from large_models.config import *
from large_models_msgs.srv import SetModel, SetString, SetInt32

from servo_controller_msgs.msg import ServosPosition, ServoPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

if os.environ["ASR_LANGUAGE"] == 'Chinese': 
    PROMPT = '''
    ##角色任务
    你是一个智能小车，负责解析用户的移动指令，提取动作和参数。

    ##动作函数库
    - "move(direction, distance, duration)"
    - direction (方向): 'forward', 'backward',  'turn_left', 'turn_right', 'shift_left', 'shift_right','stop'
    - distance (距离, 单位米): 一个数字, 如果用户没说距离则为0
    - duration (时间, 单位秒): 一个数字, 如果用户没说时间则为0

    ##要求
    1. 解析用户意图，生成包含一个或多个 move 函数调用的 action 列表。
    2. **严格遵守**：当用户指令中包含**时间**（如“前进3秒”），则**distance参数必须为0**，duration为对应的时间。
    3. **严格遵守**：当用户指令中包含**距离**（如“前进1米”），则**duration参数必须为0**，distance为对应的距离。
    4. 直接输出json结果，不要分析。
    5. 格式: {"action": ["move('forward', 1, 0)"], "response": "好的，向前1米"}
    6. 每一个动作做完之后都需要一个停止的动作,让机器人停下来: "move('forward', 1, 0)"
    

    ##任务示例
    输入：向前移动 2 秒，然后向左转 1 秒
    输出：{"action": ["move('forward', 0, 2)", "move('turn_left', 0, 1)","move('stop', 0, 1)"], "response": "收到，马上执行！"}
    输入：向前走1米
    输出：{"action": ["move('forward', 1, 0)","move('stop', 0, 1)"], "response": "好嘞，出发！"}
    '''
else:
    PROMPT = '''
    ## Role and Task
    You are an intelligent robotic car, responsible for parsing user movement commands and extracting the action and parameters.

    ## Action Function Library
    - "move(direction, distance, duration)"
    - direction (Direction): 'forward', 'backward', 'turn_left', 'turn_right', 'shift_left', 'shift_right', 'stop'
    - distance (Distance, Unit: meters): A number. If the user does not specify distance, it should be 0.
    - duration (Duration, Unit: seconds): A number. If the user does not specify time, it should be 0.

    ## Requirements
    1. Parse the user's intent and generate an 'action' list containing one or more 'move' function calls.
    2. **Strict Compliance**: When the user command includes a **duration** (e.g., "move forward for 3 seconds"), the **distance parameter must be 0**, and 'duration' should be the corresponding time.
    3. **Strict Compliance**: When the user command includes a **distance** (e.g., "move forward 1 meter"), the **duration parameter must be 0**, and 'distance' should be the corresponding distance.
    4. Output the JSON result directly. Do not provide analysis.
    5. Format: {"action": ["move('forward', 1, 0)"], "response": "OK, 1 meter forward."}
    6. A 'stop' action must be included after every movement action to ensure the robot stops, e.g., "move('forward', 1, 0)" should be followed by a stop action.
    

    ## Task Examples
    Input: Move forward for 2 seconds, then turn left for 1 second.
    Output: {"action": ["move('forward', 0, 2)", "move('turn_left', 0, 1)","move('stop', 0, 1)"], "response": "Acknowledged, executing immediately!"}
    Input: Walk forward 1 meter.
    Output: {"action": ["move('forward', 1, 0)","move('stop', 0, 1)"], "response": "On it, departing now!"}
'''

class LLMControlMove(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        
        self.action = []
        self.llm_result = ''
        self.running = True
        self.interrupt = False
        self.action_finish = False
        self.play_audio_finish = False

        self.declare_parameter('interruption', False)
        self.interruption = self.get_parameter('interruption').value
        self.declare_parameter('offline', 'false')
        self.asr_mode = os.environ.get("ASR_MODE", "online").lower()


        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1, callback_group=timer_cb_group)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()

        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        # self.awake_client.wait_for_service()
        self.set_mode_client = self.create_client(SetInt32, 'vocal_detect/set_mode')
        # self.set_mode_client.wait_for_service()
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def init_process(self):
        self.timer.cancel()

        msg = SetModel.Request()
        # msg.model = llm_model
        msg.model_type = 'llm'
        msg.model = 'qwen3:1.7b'
        msg.base_url = ollama_host
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = PROMPT
        self.send_request(self.set_prompt_client, msg)

        speech.play_audio(start_audio_path) 
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        self.get_logger().info('\033[1;32m%s\033[0m' % PROMPT)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def wakeup_callback(self, msg):
        if self.llm_result:
            self.get_logger().info('wakeup interrupt')
            self.interrupt = msg.data

    def llm_result_callback(self, msg):
        self.llm_result = msg.data

    def play_audio_finish_callback(self, msg):
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.awake_client, msg)
        # msg = SetInt32.Request()
        # msg.data = 1
        # self.send_request(self.set_mode_client, msg)
        self.play_audio_finish = msg.data
    def parse_action(self, action_str):
            """
            Helper function to parse strings in the format 'move('direction', distance, duration)'. (辅助函数，用于解析 "move('direction', distance, duration)" 格式的字符串。)
            Uses regular expressions to extract parameters, more robust. (使用正则表达式提取参数，更健壮。)
            """
            # Match content inside move(...) (匹配 move(...) 中的内容)
            match = re.search(r"move\((.*)\)", action_str)
            if not match:
                return None, 0, 0

            # Extract parameter string, e.g., "'forward', 5, 0" (提取参数字符串，例如 "'forward', 5, 0")
            params_str = match.group(1)
            
            # Split parameters (分割参数)
            params = [p.strip() for p in params_str.split(',')]
            
            # Extract and convert types (提取并转换类型)
            direction = params[0].strip("'\"") # Remove quotes from string parameter (去掉字符串参数的引号)
            distance = float(params[1])
            duration = float(params[2])
            
            return direction, distance, duration

    def process(self):

        DEFAULT_LINEAR_SPEED = 0.2  # m/s
        DEFAULT_ANGULAR_SPEED = 1.0 # rad/s 

        while self.running:
            if self.llm_result:
                msg = String()
                if 'action' in self.llm_result:  # If there is a corresponding action returned, extract and process it (如果有对应的行为返回那么就提取处理)
                    result = eval(self.llm_result[self.llm_result.find('{'):self.llm_result.find('}') + 1])
                    self.get_logger().info(str(result))
                    action_list = []
                    if 'action' in result:
                        action_list = result['action']
                    if 'response' in result:
                        response = result['response']
                    msg.data = response
                    self.tts_text_pub.publish(msg)
                    # Loop to execute action list (循环执行动作列表)
                    for action_str in action_list:
                        direction, distance, duration = self.parse_action(action_str)
                        
                        if direction is None:
                            continue # Parsing failed, skip this action (解析失败，跳过这个动作)

                        twist_msg = Twist()
                        final_duration = 0

                        # Set speed based on direction (根据方向设置速度)
                        if direction == 'forward':
                            twist_msg.linear.x = DEFAULT_LINEAR_SPEED
                        elif direction == 'backward':
                            twist_msg.linear.x = -DEFAULT_LINEAR_SPEED
                        elif direction == 'shift_left':
                            twist_msg.linear.y = DEFAULT_LINEAR_SPEED
                        elif direction == 'shift_right':
                            twist_msg.linear.y = -DEFAULT_LINEAR_SPEED
                        elif direction == 'turn_left':
                            twist_msg.angular.z = DEFAULT_ANGULAR_SPEED
                        elif direction == 'turn_right':
                            twist_msg.angular.z = -DEFAULT_ANGULAR_SPEED
                        
                        if distance > 0:
                            # Prioritize using distance to calculate time (优先使用距离计算时间)
                            final_duration = distance / (DEFAULT_LINEAR_SPEED if 'turn' not in direction else DEFAULT_ANGULAR_SPEED)
                        else:
                            # Otherwise use specified duration (否则使用指定的持续时间)
                            final_duration = duration

                        # Publish movement command and wait (发布移动指令并等待)
                        if final_duration > 0:
                            self.get_logger().info(f"Action: {direction}, Publishing velocity, Duration: {final_duration:.2f}s")
                            self.mecanum_pub.publish(twist_msg)
                            time.sleep(final_duration)

                        # Check if interruption is needed (检查是否需要中断)
                        if self.interrupt:
                            # self.get_logger().info("Action sequence interrupted.")
                            self.interrupt = False
                            break # Break out of for loop (跳出 for 循环)

                    # After all actions are executed or interrupted, send stop command (所有动作执行完毕或被中断后，发送停止指令)
                    self.mecanum_pub.publish(Twist())
                else:  # No corresponding action, only answer (没有对应的行为，只回答)
                    response = self.llm_result
                    msg.data = response
                    self.tts_text_pub.publish(msg)
                self.action_finish = True 
                self.llm_result = ''
            else:
                time.sleep(0.01)
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                if self.interruption:
                    msg = SetInt32.Request()
                    msg.data = 2
                    self.send_request(self.set_mode_client, msg)
        rclpy.shutdown()

def main():
    node = LLMControlMove('llm_control_move')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
