#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2025/03/06
import os
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
你是一辆智能小车，可以通过 x 方向和 y 方向控制线速度,单位m/s，并通过 z 方向控制角速度,单位rad/s，t控制时间单位s。需要根据输入的内容，生成对应的指令。

##要求
1.确保速度范围正确：
线速度：x, y ∈ [-1.0, 1.0]（负值表示反方向）
角速度：z ∈ [-1.0, 1.0]（向右边为负，向左边为正）
2.顺序执行多个动作，输出一个 包含多个移动指令的 action 列表，仅在最后一个动作后添加 [0.0, 0.0, 0.0, 0.0] 以确保小车停止。
3.x和y默认为0.2, z默认为1, t默认为2，左转或者右转默认需提供x的线速度方向和大小。 
4.为每个动作序列编织一句精炼（5至10字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
5.直接输出json结果，不要分析，不要输出多余内容。
6.格式：
{  
  "action": [[x1, y1, z1, t1], [x2, y2, z2, t2], ..., [0.0, 0.0, 0.0, 0.0]],  
  "response": "xx"  
}  
7.很强的数学计算能力

##特别注意
- "action"键下承载一个按执行顺序排列的函数名称字符串数组，当找不到对应动作函数时action输出[]。 
- "response"键则配以精心构思的简短回复，完美贴合上述字数与风格要求。 

##任务示例
输入：向前移动 2 秒，然后向左转 1 秒
输出：{"action": [[0.2, 0.0, 0.0, 2.0], [0.2, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], "response": "前进 2 秒，然后向左转 1 秒，出发！"}
输入：向前走1米
输出：{"action": [[0.2, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0]], "response": "好嘞"}
    '''
else:
    PROMPT = '''
##Role Task
You are a smart car that can control the linear speed in the x and y directions (m/s), and the angular speed in the z direction (rad/s), and the time in t (s). You need to generate corresponding instructions based on the input content.

##Requirements
1. Ensure that the speed range is correct:
Linear speed: x, y ∈ [-1.0, 1.0] (negative value indicates the opposite direction)
Angular speed: z ∈ [-1.0, 1.0] (negative to the right, positive to the left)
2. Execute multiple actions in sequence and output an action list containing multiple movement instructions. Only add [0.0, 0.0, 0.0, 0.0] after the last action to ensure that the car stops.
3. The default values ​​of x and y are 0.2, z is 1, and t is 2. Turn left or right by default. The linear speed direction and magnitude of x are required.
4. For each action sequence, weave a concise (5 to 10 words), humorous and ever-changing feedback message to make the communication process interesting.
5. Directly output the json result, do not analyze, and do not output redundant content.
6. Format:
{
"action": [[x1, y1, z1, t1], [x2, y2, z2, t2], ..., [0.0, 0.0, 0.0, 0.0]],
"response": "xx"
}
7. Strong mathematical calculation ability

##Special attention
- The "action" key carries an array of function name strings arranged in execution order. When the corresponding action function cannot be found, action outputs [].
- The "response" key is equipped with a carefully conceived short reply, which perfectly meets the above word count and style requirements.

##Task example
Input: Move forward for 2 seconds, then turn left for 1 second
Output: {"action": [[0.2, 0.0, 0.0, 2.0], [0.2, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], "response": "Move forward for 2 seconds, then turn left for 1 second, let's go!"}
Input: Move forward 1 meter
Output: {"action": [[0.2, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0]], "response": "Okay"}
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
        self.asr_mode = os.environ.get("ASR_MODE", "online").lower()
        
        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1, callback_group=timer_cb_group)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()

        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.set_mode_client = self.create_client(SetInt32, 'vocal_detect/set_mode')
        self.set_mode_client.wait_for_service()
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def init_process(self):
        self.timer.cancel()

        msg = SetModel.Request()
        # msg.model = 'qwen-plus-latest'
        msg.model = llm_model
        msg.model_type = 'llm'
        msg.api_key = api_key 
        msg.base_url = base_url
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

    def process(self):
        while self.running:
            if self.llm_result:
                msg = String()
                if 'action' in self.llm_result:  # If there is a corresponding action returned, then extract and process it (如果有对应的行为返回那么就提取处理)
                    result = eval(self.llm_result[self.llm_result.find('{'):self.llm_result.find('}') + 1])
                    self.get_logger().info(str(result))
                    action_list = []
                    if 'action' in result:
                        action_list = result['action']
                    if 'response' in result:
                        response = result['response']
                    msg.data = response
                    self.tts_text_pub.publish(msg)
                    for i in action_list:
                        msg = Twist()
                        msg.linear.x = float(i[0])
                        msg.linear.y = float(i[1])
                        msg.angular.z = float(i[2])
                        self.mecanum_pub.publish(msg)
                        time.sleep(i[3])
                        if self.interrupt:
                            self.interrupt = False
                            self.mecanum_pub.publish(Twist())
                            break
                else:  # No corresponding action, only respond (没有对应的行为，只回答）
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
