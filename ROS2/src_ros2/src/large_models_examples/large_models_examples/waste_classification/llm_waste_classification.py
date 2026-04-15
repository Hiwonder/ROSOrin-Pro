#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import re
import os
import time
import json
import rclpy
import threading
from config import *
from speech import speech
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty

from large_models.config import *
from large_models_msgs.srv import SetModel, SetString

from interfaces.srv import SetStringBool
from interfaces.srv import SetStringList
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

	
if os.environ["ASR_LANGUAGE"] == 'Chinese':
    PROMPT = '''
#角色任务
你是一款智能机械臂，需要根据输入的内容，生成对应的指令。

##要求与限制
1.根据输入的动作内容，在动作函数库中找到对应的指令，并输出对应的指令。
	2.为动作序列编织一句精炼（10至30字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
	3.垃圾分为四类：厨余垃圾（food_waste），有害垃圾（hazardous_waste），可回收垃圾（recyclable_waste），其他垃圾（residual_waste）。
4.直接输出json结果，不要分析，不要输出多余内容。
5.格式：{"action": "xx", "response": "xx"}

##结构要求：
- "action"键下承载一个函数名称字符串数组，当找不到对应动作函数时action输出""。 
- "response"键则配以精心构思的简短回复，完美贴合上述字数与风格要求。 

##实施原则 
- 在最终输出前，实施全面校验，确保回复不仅格式合规，而且内容充满意趣，无一遗漏上述规范。

##动作函数库
	- 分拣不同垃圾：garbage_classification('food_waste')
	- 停止：garbage_classification('')

##任务示例
输入：将厨余垃圾拿走
输出：{"action": "garbage_classification('food_waste')", "response": "收到，我马上收拾"}
输入：把香蕉皮拿走
输出：{"action": "garbage_classification('food_waste')", "response": "好的，香蕉皮属于厨余垃圾"}
输入：清理有害垃圾
输出：{"action": "garbage_classification('hazardous_waste')", "response": "有害垃圾必须清除"}
输入：将可回收垃圾找出来
输出：{"action": "garbage_classification('recyclable_waste')", "response": "这可难不倒我"}
输入：分拣其他垃圾
输出：{"action": "garbage_classification('residual_waste')", "response": "收到，收到"}
输入：先把厨余垃圾拿走，再把有害垃圾清理了
输出：{"action": "garbage_classification('food_waste'，'hazardous_waste')", "response": "垃圾清理，马上执行"}
输入：只留下可回收垃圾，把其都拿走
输出：{"action": "garbage_classification('food_waste‘，’hazardous_waste'，"residual_waste')", "response": "没问题"}
'''
else:
    PROMPT = '''
#Role
You are an intelligent robot arm that needs to generate corresponding instructions based on the input content.

##Requirements and Restrictions
1. According to the input action content, find the corresponding instructions in the action function library and output the corresponding instructions.
2. Weave a concise (10 to 30 words), humorous and ever-changing feedback information for the action sequence to make the communication process interesting.
3. Garbage is divided into four categories: food waste (food_waste), hazardous waste (hazardous_waste), recyclable waste (recyclable_waste), and other waste (residual_waste).
4. Directly output the json result, do not analyze it, and do not output redundant content.
5. Format: {"action": "xx", "response": "xx"}

##Structural Requirements:
- The "action" key carries a function name string array. When the corresponding action function cannot be found, action outputs "".
- The "response" key is accompanied by a carefully conceived short reply, which perfectly meets the above word count and style requirements.

##Implementation Principles 
- Before final output, implement comprehensive verification to ensure that the response is not only formatted in compliance, but also interesting and without missing any of the above specifications.

##Action function library
- Sort different garbage: garbage_classification('food_waste')
- Stop: garbage_classification('')

##Example
Input: Take away the food waste
Output: {"action": "garbage_classification('food_waste')", "response": "Got it, I'll clean it up right away"}
Input: Take away the banana peel
Output: {"action": "garbage_classification('food_waste')", "response": "Okay, banana peels belong to food waste"}
Input: Clean up hazardous waste
Output: {"action": "garbage_classification('hazardous_waste')", "response": "Hazardous waste must be removed"}
Input: Find recyclable waste
Output: {"action": "garbage_classification('recyclable_waste')", "response": "This is not a problem for me"}
Input: Sort other garbage
Output: {"action": "garbage_classification('residual_waste')", "response": "Received, received"}
Input: Take away the food waste first, then clean up the hazardous waste
Output: {"action": "garbage_classification('food_waste'，'hazardous_waste')", "response": "Garbage cleanup, execute immediately"}
Input: Only recyclable garbage is left, take away everything else
Output: {"action": "garbage_classification('food_waste‘，'hazardous_waste'，"residual_waste')", "response": "No problem"}
    '''

ALL_TARGETS = [
    # 类别目标（顺序固定）
    'food_waste', 'hazardous_waste', 'recyclable_waste', 'residual_waste',
    # 物品目标
    'BananaPeel', 'BrokenBones', 'Ketchup',
    'Marker', 'OralLiquidBottle', 'StorageBattery',
    'PlasticBottle', 'Toothbrush', 'Umbrella',
    'Plate', 'CigaretteEnd', 'DisposableChopsticks',
]

class LLMColorSorting(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        
        self.action = []
        self.llm_result = ''
        self.action_finish = False
        self.play_audio_finish = False
        # self.llm_result = '{"action": "color_sorting(\'blue\', \'red\', \'green\', \'tag_1\')", "response": "ok！"}'
        self.running = True
        
        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1)
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()

        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        self.enter_client = self.create_client(Trigger, 'waste_classification/enter')
        self.enter_client.wait_for_service()
        self.start_client = self.create_client(SetBool, 'waste_classification/enable_transport')
        self.start_client.wait_for_service()

        self.set_target_client = self.create_client(SetStringList, 'waste_classification/set_target')
        self.set_target_client.wait_for_service()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def init_process(self):
        self.timer.cancel()
        
        msg = SetModel.Request()
        msg.model = llm_model
        msg.model_type = 'llm'
        msg.api_key = api_key 
        msg.base_url = base_url
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = PROMPT
        self.send_request(self.set_prompt_client, msg)
       
        self.send_request(self.enter_client, Trigger.Request())
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
        if msg.data and self.llm_result:
            self.send_request(self.enter_client, Trigger.Request())

    def llm_result_callback(self, msg):
        self.llm_result = msg.data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def process(self):
        while self.running:
            if self.llm_result:
                msg = String()
                if 'action' in self.llm_result: # If a corresponding action is returned, extract it and proceed with processing. 如果有对应的行为返回那么就提取处理
                    result = eval(self.llm_result[self.llm_result.find('{'):self.llm_result.find('}') + 1])
                    # result = json.loads(self.llm_result)
                    if 'action' in result:
                        text = result['action']
                        # Use regular expressions to extract all strings inside. 使用正则表达式提取括号中的所有字符串
                        pattern = r"\((['\"][^'\"]+['\"](?:\s*,\s*['\"][^'\"]+['\"])*)\)" 
                        #  Use re.search to find the matching result. 使用re.search找到匹配的结果
                        match = re.search(pattern, text)
                        self.get_logger().info(f'{match}')
                        # Extract the result. 提取结果
                        if match:
                            # Retrieve all argument parts (the content inside the parentheses). 获取所有的参数部分（括号内的内容）
                            params = match.group(1)
                            # Use regular expressions to split the string arguments. 使用正则拆分出各个字符串参数
                            target_list = re.findall(r"['\"]([^'\"]+)['\"]", params)
                            self.get_logger().info(str(target_list))
                            waste_msg = SetStringBool.Request()
                            # for i in ALL_TARGETS:
                            #     waste_msg.data_str = i
                            #     if i in target_list:
                            #         waste_msg.data_bool = True
                            #     else:
                            #         waste_msg.data_bool = False
                            #     self.send_request(self.set_target_client, waste_msg)
                            
                            waste_msg = SetStringList.Request()
                            waste_msg.data = target_list
                            self.send_request(self.set_target_client, waste_msg)
                            self.get_logger().info(f'{target_list}')
                            # Enable sorting. 开启分拣
                            start_msg = SetBool.Request()
                            start_msg.data = True 
                            self.send_request(self.start_client, start_msg)
                    if 'response' in result:
                        msg.data = result['response']
                else: # No corresponding action, just respond. 没有对应的行为，只回答
                    msg.data = self.llm_result
                self.tts_text_pub.publish(msg)
                self.action_finish = True
                self.llm_result = ''
            else:
                time.sleep(0.01)
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
        rclpy.shutdown()

def main():
    node = LLMColorSorting('llm_color_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
