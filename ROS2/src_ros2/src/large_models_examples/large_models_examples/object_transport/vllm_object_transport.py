#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import cv2
import math
import json
import time
import queue
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from std_srvs.srv import Trigger, SetBool, Empty
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from speech import speech
from large_models.config import *
from large_models_msgs.srv import SetString, SetModel, SetBox, SetContent

if os.environ["ASR_LANGUAGE"] == 'Chinese':
    LLM_PROMPT = '''
#角色任务
作为语言专家，你能深刻的洞悉用户的指令。

##技能细则
* 分析用户指令，将一次抓取和一次放置作为一个动作
* 有较强的逻辑能力
* 将物体放回初始位置时需要考虑现在物体的关系，然后决定搬运的先后顺序
* 只有移动过的物体才可以放回初始位置
* 方位一般有左，右， 上，前面，后面，如果没有指定数值，当左时为[0.0,0.11]，当右时为[0.0,-0.11]，当前时为[0.06,0.0]，当后时为[-0.06,0.0]，当上时为[0.0,0.0]

##要求与限制
1.根据输入的内容，在函数库中找到对应的指令，并输出对应的指令。
2.多个动作需要放到列表里
3.搬运分为拿起和放下两个阶段
4.直接输出json格式的数据，不要分析，不要输出多余内容。
5.复原时, 如果指令中有搬运的动作，那么需要将搬运的指令返回, 否则返回""
6.【重要】复原时，必须回顾整句指令，提取出**所有**曾经被移动过的不同物体。无论最后操作的是哪个，只要物体位置被改变过，就必须生成对应的 restore 指令。
7.格式：{"action": "xx"}

##可用函数与操作：
* 拿起物体：[["将红色物体拿起来", "pick"]]
* 放到指定物体的具体方位上：[["放到蓝色上", "place", [0.0, 0.0]]]
* 记住当前物体位置：[["remember"]]
* 将指定物体放回初始位置: [["将蓝色放回初始位置", "restore"]]
* 将一个物体放到另一个物体的具体方位上: [["将蓝色放到红色上", "pick_and_place", [0.0, 0.0]]]
* 复原: [["", "recovery"]]

##示例：
输入：将蓝色放到黄色上，将绿色放到蓝色上，将物体放回初始位置，只需要放回去的步骤
输出：{"action": [["将绿色放回初始位置", "restore"], ["将蓝色放回初始位置", "restore"]]}
输入：将蓝色放到红色上，将物体放回初始位置，只需要放回去的步骤
输出：{"action": [["将红色放回初始位置", "restore"]]}
输入：记住，将蓝色放到红色右边，复原
输出：{"action": [["remember"], ["将蓝色放到红色右边", "pick_and_place", [0.0, -0.06]], ["将蓝色放到红色右边", "recovery"]]}
输入：记住，将蓝色放到黄色左边8cm的地方，将绿色放到蓝色右边7cm，复原
输出：{"action": [["remember"], ["将蓝色放到黄色左边8cm的地方", "pick_and_place", [0.0, 0.08]], ["将绿色放到蓝色右边7cm", "pick_and_place", [0.0, -0.07]],["将蓝色放到黄色左边8cm的地方，将绿色放到蓝色右边7cm", "recovery"]]}
输入：将蓝色放到红色上
输出：{"action": [["将蓝色放到红色上"，"pick_and_place", [0.0, 0.0]]]}
输入：将蓝色拿起来，放到绿色上
输出：{"action": [["将蓝色放到绿色上", "pick_and_place", [0.0, 0.0]]]}
输入：将红色拿起来
输出：{"action": [["将红色拿起来", "pick"]]}
输入：放到红色上
输出：{"action": [["放到红色上", "place", [0.0, 0.0]]]}
    '''

    VLLM_PROMPT1 = '''
你作为智能管家机器人，可以抓取各种东西， 你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出。
## 1. 理解用户指令
我会给你一句话，从我的话中提取「物体名称」, **object对应的name要用英文表示**, **不要输出没有提及到的物体**
## 2. 理解图片
我会给你一张图, 从这张图中找到「物体名称」对应物体的左上角和右下角的像素坐标, **不要输出没有提及到的物体**
【特别注意】： 涉及到初始位置时place的name为"", "xyxy"为[]。当pick或者place没有相应物体时，name为""， "xyxy"为[]
## 输出格式（请仅输出以下内容，不要说任何多余的话
{
 "pick": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]},
 "place": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]},
 "response": "人性化的回复，回复可以简洁点和你接下来的行为有关, 用中文"
}
## 示例
输入：放到红色上
输出：{
 "pick": {"object": "", "xyxy": ""},
 "place": {"object": "red", "xyxy": [xmin, ymin, xmax, ymax]},
 "response": "人性化的回复，回复可以简洁点和你接下来的行为有关, 用中文"
}
    '''

    VLLM_PROMPT2 = '''
你是一个图像识别专家，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出。
## 1. 理解图片
我会给你一张图片，如果图中有红绿蓝三种颜色方块，那么请识别图中红绿蓝三种颜色方块的位置 
## 2. 物体定位 - position
检测到每一个物体的位置，根据你的知识和记忆，看清物体的样子，识别每个物体的具体名称, **请输出全部的物体不要遗漏**
【特别注意】： name要用英文表示, 且能突出物体的特征。
## 输出格式（请仅输出以下内容，不要说任何多余的话
{
 "name": [xmin, ymin, xmax, ymax],
 "name": [xmin, ymin, xmax, ymax],
}
    '''
else:
    LLM_PROMPT = '''
#Role
As a language expert, you can deeply understand the user's instructions.

##Skills
* Analyze user instructions and treat a grab and a placement as an action
* Have strong logical ability
* When putting an object back to its original position, you need to consider the current relationship of the objects and then decide the order of transportation
* Only objects that have been moved can be put back to their original positions
* The directions are generally left, right, top, front, and back. If no value is specified, when left, it is [0.0,0.06], when right, it is [0.0,-0.06], when current, it is [0.06,0.0], when back, it is [-0.06,0.0], and when top, it is [0.0,0.0]

##Requirements and Restrictions
1. According to the input content, find the corresponding instructions in the function library and output the corresponding instructions.
2. Multiple actions need to be put in a list
3. Transportation is divided into two stages: picking up and putting down
4. Directly output data in json format, do not analyze, and do not output redundant content.
5. When restoring, if there is a move action in the instruction, then the move instruction needs to be returned, otherwise it returns ""
6. Format: {"action": "xx"}

##Available functions and operations:
* Pick up an object: [["Pick up the red object", "pick"]]
* Put it in the specific position of the specified object: [["Put it on the blue", "place", [0.0, 0.0]]]
* Remember the current object position: [["remember"]]
* Put the specified object back to the initial position: [["Put the blue back to the initial position", "restore"]]
* Put an object in the specific position of another object: [["Put the blue on the red", "pick_and_place", [0.0, 0.0]]]
* Recovery: [["", "recovery"]]

##Example:
Input: Put blue on yellow, put green on blue, put the object back to the initial position, only the steps of putting it back are needed
Output: {"action": [["Put green back to its original position", "restore"], ["Put blue back to its original position", "restore"]]}
Input: Put blue on red, put the object back to its original position, only need to put it back
Output: {"action": [["Put red back to its original position", "restore"]]}
Input: Remember, put blue to the right of red, restore
Output: {"action": [["remember"], ["Put blue to the right of red", "pick_and_place", [0.0, -0.06]], ["Put blue to the right of red", "recovery"]]}
Input: Remember, put blue 8cm to the left of yellow, put green 7cm to the right of blue, restore
Output: {"action": [["remember"], ["Put blue 8cm to the left of yellow", "pick_and_place", [0.0, 0.08]], ["Put green 7cm to the right of blue", "pick_and_place", [0.0, -0.07]],["Put blue 8cm to the left of yellow, and green 7cm to the right of blue", "recovery"]]}
Input: Put blue on red
Output: {"action": [["Put blue on red", "pick_and_place", [0.0, 0.0]]]}
Input: Pick up blue and put it on green
Output: {"action": [["Put blue on green", "pick_and_place", [0.0, 0.0]]]}
Input: Pick up red
Output: {"action": [["Pick up red", "pick"]]}
Input: Put on red
Output: {"action": [["Put on red", "place", [0.0, 0.0]]]}
    '''
    
    VLLM_PROMPT1 = '''
**Role
You are an intelligent housekeeper robot capable of identifying and grasping various objects. Your task is to perform object detection and precise localization based on the image provided by the user, and return results in the specified Output Format.

**Task
1. Understand User Instruction
You will receive a sentence. Extract the mentioned object name from it.
The object's name must be expressed in English
Do not output any object not explicitly mentioned

2. Understand the Image
You will receive an image. Find the pixel coordinates of the top-left and bottom-right corners of the object matching the extracted name.
Do not detect or include unmentioned objects

**Special Notes
If the instruction refers only to the initial position, then place.object is "" and place.xyxy is []
If pick or place has no corresponding object, then "object" should be "" and "xyxy" should be []

**Output Format (Only return the JSON content below with no extra text, the coordinates (xmin, ymin, xmax, ymax) must be normalized to the range [0, 1]):
{
  "pick": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]},
  "place": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]
  "response": "A human-like reply, short and related to your next action"

**Example
Input: put it on the red one
Output:
{
  "pick": {"object": "", "xyxy": []},
  "place": {"object": "red", "xyxy": [xmin, ymin, xmax, ymax]},
  "response": "Got it. Placing the object on the red one now."
}
    '''

    VLLM_PROMPT2 = '''
**Role
You are an image recognition expert. Your task is to perform accurate object detection and localization based on the image provided by the user, and return the result in the specified Output Format.

**Task
1. Image Understanding
You will receive an image. If it contains red, green, and blue square blocks, identify and locate all of them in the image.

2. Object Localization - position
For each detected object, determine its location by returning the pixel coordinates of the top-left and bottom-right corners in the format [xmin, ymin, xmax, ymax].
Use your knowledge and visual recognition ability to determine a descriptive and specific English name for each object.

**Special Notes
Output all detected objects, do not miss any.
Use English names that highlight the object's characteristics.

**Output Format (Only output in the format below. Do not include any additional text, the coordinates (xmin, ymin, xmax, ymax) must be normalized to the range [0, 1])
{
  "object_name": [xmin, ymin, xmax, ymax],
  "object_name": [xmin, ymin, xmax, ymax]
}
    '''


class VLLMObjectTransport(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.llm_result = ''
        # self.vllm_result = '''json{"object":"香蕉","xyxy":[243, 441, 842, 552], "response": "香蕉"}'''
        self.running = True
        self.transport_finished = True
        self.lock = threading.RLock()
        self.client = speech.OpenAIAPI(api_key, base_url)
        self.language = os.environ["ASR_LANGUAGE"]
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.asr_pub = self.create_publisher(String, '/vocal_detect/asr_result', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'object_transport/transport_finished', self.transport_finished_callback, 1)


        self.get_logger().info("Waiting for service 'vocal_detect/enable_wakeup'...")
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.get_logger().info("Service 'vocal_detect/enable_wakeup' found.")

        self.get_logger().info("Waiting for service 'agent_process/set_model'...")
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.get_logger().info("Service 'agent_process/set_model' found.")


        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()

        self.set_vllm_content_client = self.create_client(SetContent, 'agent_process/set_vllm_content')
        self.set_vllm_content_client.wait_for_service()

        self.record_chat_client = self.create_client(SetBool, 'agent_process/record_chat')
        self.record_chat_client.wait_for_service()

        self.get_chat_client = self.create_client(Trigger, 'agent_process/get_chat')
        self.get_chat_client.wait_for_service()

        self.clear_chat_client = self.create_client(Empty, 'agent_process/clear_chat')
        self.clear_chat_client.wait_for_service()

        self.enter_client = self.create_client(Trigger, '/object_transport/enter')
        self.enter_client.wait_for_service()

        self.start_client = self.create_client(SetBool, '/object_transport/enable_transport')
        self.start_client.wait_for_service()

        self.set_pick_client = self.create_client(SetBox, 'object_transport/set_pick_position')
        self.set_pick_client.wait_for_service()

        self.set_place_client = self.create_client(SetBox, 'object_transport/set_place_position')
        self.set_place_client.wait_for_service()

        self.record_position_client = self.create_client(SetBox, 'object_transport/record_position')
        self.record_position_client.wait_for_service()

        timer_cb_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def init_process(self):
        self.timer.cancel()

        msg = SetModel.Request()
        msg.model_type = 'llm'
        if self.language == 'Chinese':
            msg.model = 'qwen3-max'
            msg.api_key = api_key 
            msg.base_url = base_url
        else:
            msg.model =  'qwen/qwen3-max'
            msg.api_key = vllm_api_key 
            msg.base_url = vllm_base_url
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = LLM_PROMPT
        self.send_request(self.set_prompt_client, msg)
        
        msg = Empty.Request()
        self.send_request(self.clear_chat_client, msg)

        init_finish = self.create_client(Empty, 'object_transport/init_finish')
        init_finish.wait_for_service()
        self.send_request(self.enter_client, Trigger.Request())
        speech.play_audio(start_audio_path)
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        self.get_logger().info('\033[1;32m%s\033[0m' % LLM_PROMPT)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def llm_result_callback(self, msg):
        self.llm_result = msg.data

    def remember_position(self):
        msg = SetContent.Request()
        if self.language == 'Chinese':
            msg.api_key = aliyun_api_key
            msg.base_url = aliyun_base_url
            msg.model = aliyun_vllm_model
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        msg.prompt = VLLM_PROMPT2 
        # self.get_logger().info(f'vllm_result22222222222222222222')
        vllm_result = self.send_request(self.set_vllm_content_client, msg)
        # self.get_logger().info(f'vllm_result: {vllm_result}')
        message = vllm_result.message
        if message.startswith("```") and message.endswith("```"):
            message = message.strip("```").replace("json\n", "").strip()
        initial_position = json.loads(message)
        
        # self.get_logger().info(f'initial_position: {initial_position}')
        for i in initial_position:
            msg = SetBox.Request()
            msg.label = i
            xyxy = initial_position[i]
            if self.language == 'Chinese':
                # msg.box = self.client.data_process(xyxy, 640, 480)
                msg.box = [int(x) for x in xyxy]  #千问
            else:
                msg.box = [int(xyxy[0] * 640), int(xyxy[1] * 480), int(xyxy[2] * 640), int(xyxy[3] * 480)]
            self.send_request(self.record_position_client, msg)
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.start_client, msg)

        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.record_chat_client, msg)

    def recovery(self, text):
        if not text:
            msg = Trigger.Request()
            res = self.send_request(self.get_chat_client, msg)
            # self.get_logger().info(str(res.message)) 
            text = res.message.rsplit(",", 1)[0] if "," in res.message else res.message
            msg = SetBool.Request()
            msg.data = False
            self.send_request(self.record_chat_client, msg)
            msg = Empty.Request()
            self.send_request(self.clear_chat_client, msg)
        msg = String()
        if self.language == 'Chinese':
            msg.data = '{}, 只需要放回去的步骤'.format(text)
        else:
            msg.data = '{}, only the steps for putting it back'.format(text)
        self.get_logger().info('\033[1;32m%s\033[0m' % msg.data)
        self.asr_pub.publish(msg)

    def object_restore(self, query): 
        msg = SetContent.Request()
        if self.language == 'Chinese':
            msg.api_key = aliyun_api_key
            msg.base_url = aliyun_base_url
            msg.model = aliyun_vllm_model
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        msg.prompt = VLLM_PROMPT1
        msg.query = query
        vllm_result = self.send_request(self.set_vllm_content_client, msg)
        self.get_logger().info(f'vllm_result: {vllm_result}')
        message = vllm_result.message
        if message.startswith("```") and message.endswith("```"):
            message = message.strip("```").replace("json\n", "").strip()
        position = json.loads(message)
        
        msg = SetBox.Request()
        msg.label = position['pick']['object']
        xyxy = position['pick']['xyxy']
        if self.language == 'Chinese':
            # msg.box = self.client.data_process(xyxy, 640, 480)
            msg.box = [int(x) for x in xyxy]  #千问
        else:
            msg.box = [int(xyxy[0] * 640), int(xyxy[1] * 480), int(xyxy[2] * 640), int(xyxy[3] * 480)]
        self.send_request(self.set_pick_client, msg)
        msg = SetBox.Request()
        msg.label = position['pick']['object']
        self.send_request(self.set_place_client, msg)
        # self.get_logger().info(f'msg: {msg}')

        # Enable sorting. 开启分拣
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.start_client, msg)
        self.transport_finished = False
       
        msg = Empty.Request()
        self.send_request(self.clear_chat_client, msg)

        msg = String()
        msg.data = position['response']
        self.tts_text_pub.publish(msg)

    def get_object_position(self, query, offset=None):
        msg = SetContent.Request()
        if self.language == 'Chinese':
            msg.api_key = aliyun_api_key
            msg.base_url = aliyun_base_url
            msg.model = aliyun_vllm_model
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        msg.prompt = VLLM_PROMPT1
        msg.query = query
        # self.get_logger().info(f'vllm_result11111111111111111111111')
        vllm_result = self.send_request(self.set_vllm_content_client, msg)
        self.get_logger().info(f'vllm_result: {vllm_result}')
        message = vllm_result.message
        if message.startswith("```") and message.endswith("```"):
            message = message.strip("```").replace("json\n", "").strip()
        position = json.loads(message)
        # self.get_logger().info(f'position: {position}') 

        # msg.label = label
        xyxy = position['pick']['xyxy']
        self.get_logger().info(f'pick xyxy: {xyxy}')
        if xyxy:
            msg = SetBox.Request()
            if self.language == 'Chinese':
                # msg.box = self.client.data_process(xyxy, 640, 480) #阶跃星辰
                msg.box = [int(x) for x in xyxy]  #千问
            else:
                msg.box = [int(xyxy[0] * 640), int(xyxy[1] * 480), int(xyxy[2] * 640), int(xyxy[3] * 480)]
            # msg.box = position['pick']['xyxy']
            self.send_request(self.set_pick_client, msg)
        xyxy = position['place']['xyxy']
        # self.get_logger().info(f'place xyxy: {xyxy}')
        if xyxy:
            msg = SetBox.Request()
            # msg.label = label
            if self.language == 'Chinese':
                # msg.box = self.client.data_process(xyxy, 640, 480) #阶跃星辰
                msg.box = [int(x) for x in xyxy]  #千问

            else:
                msg.box = [int(xyxy[0] * 640), int(xyxy[1] * 480), int(xyxy[2] * 640), int(xyxy[3] * 480)]
            # msg.box = position['place']['xyxy']
            msg.offset = offset   #例如: 放在盘子左边的偏移量
            self.send_request(self.set_place_client, msg)

        # Enable sorting. 开启分拣
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.start_client, msg)
        self.transport_finished = False

        msg = String()
        msg.data = position['response']
        self.tts_text_pub.publish(msg)

    def transport_finished_callback(self, msg):
        self.transport_finished = msg.data

    def process(self):
        action_list = []
        action_finish = False
        while self.running:
            if self.llm_result:
                try:
                    self.get_logger().info(f'llm_result: {self.llm_result}')
                    if 'action' in self.llm_result: # 如果有对应的行为返回那么就提取处理
                        if self.llm_result.startswith("```") and self.llm_result.endswith("```"):
                            self.llm_result = self.llm_result.strip("```").replace("json\n", "").strip()
                        action_list = json.loads(self.llm_result)['action']
                        action_finish = False
                    else:
                        action_finish = True
                except BaseException as e:
                    print(e)
                self.llm_result = ''
            else:
                if action_list:
                    # self.get_logger().info(f'action_list33333333:{action_list}')
                    # self.get_logger().info(f'{self.transport_finished}')

                    if self.transport_finished:
                        action = action_list[0]
                        if 'remember' in action:
                            self.remember_position()
                        elif 'recovery' in action:
                            self.recovery(action[0])
                        elif 'pick' in action:
                            self.get_object_position(action[0])
                        elif 'place' in action:
                            self.get_object_position(action[0], action[-1])
                        elif 'pick_and_place' in action:
                            self.get_object_position(action[0], action[-1])
                        elif 'restore' in action:
                            self.object_restore(action[0])
                        del action_list[0]
                        if not action_list:
                            action_finish = True
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.02)
            if action_finish:
                action_finish = False
                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
        rclpy.shutdown()

def main():
    node = VLLMObjectTransport('vllm_object_transport')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
