#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import cv2
import json
import queue
import rclpy
import threading
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from speech import speech
from large_models.config import *
from large_models_msgs.srv import SetModel, SetString, SetInt32
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition, ServoPosition

VLLM_PROMPT = '''
'''

display_size = [int(640*6/4), int(480*6/4)]
class VLLMWithCamera(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.image_queue = queue.Queue(maxsize=2)
        self.set_above = False
        self.vllm_result = ''
        self.running = True
        self.action_finish = False
        self.play_audio_finish = False
        self.bridge = CvBridge()
        self.client = speech.OpenAIAPI(api_key, base_url)
        self.declare_parameter('camera_topic', '/depth_cam/rgb0/image_raw')
        camera_topic = self.get_parameter('camera_topic').value
        
        self.declare_parameter('interruption', False)
        self.interruption = self.get_parameter('interruption').value

        timer_cb_group = ReentrantCallbackGroup()
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        set_servo_position(self.joints_pub, 1, ((10, 500), (5, 500), (4, 150), (3, 50), (2, 765), (1, 500)))
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.create_subscription(Image, camera_topic, self.image_callback, 1)
        self.create_subscription(String, 'agent_process/result', self.vllm_result_callback, 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.set_mode_client = self.create_client(SetInt32, 'vocal_detect/set_mode')
        self.set_mode_client.wait_for_service()
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def init_process(self):
        self.timer.cancel()
        
        msg = SetModel.Request()
        msg.model = vllm_model
        msg.model_type = 'vllm'
        if os.environ['ASR_LANGUAGE'] == 'Chinese':
            msg.model = stepfun_vllm_model
            msg.api_key = stepfun_api_key
            msg.base_url = stepfun_base_url
        else:
            msg.model = vllm_model
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = VLLM_PROMPT
        self.send_request(self.set_prompt_client, msg)

        set_servo_position(self.joints_pub, 1.0,
                           ((1, 500), (2, 645), (3, 135), (4, 80), (5, 500), (10, 220)))  # Set the robotic arm to its initial position（设置机械臂初始位置）
        speech.play_audio(start_audio_path)
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def vllm_result_callback(self, msg):
        self.vllm_result = msg.data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def process(self):
        # box = []
        while self.running:
            image = self.image_queue.get(block=True)
            if self.vllm_result:
                msg = String()
                # self.get_logger().info('vllm_result: %s' % self.vllm_result)
                msg.data = self.vllm_result
                self.tts_text_pub.publish(msg)
                # if self.vllm_result.startswith("```") and self.vllm_result.endswith("```"):
                    # self.vllm_result = self.vllm_result.strip("```").replace("json\n", "").strip()
                # self.vllm_result = json.loads(self.vllm_result)
                # box = self.vllm_result['xyxy']
                # if box:
                    # self.get_logger().info('box: %s' % str(box))
                    # box = [int(box[0] * 640), int(box[1] * 480), int(box[2] * 640), int(box[3] * 480)]
                    # box[0] = int(box[0] / 640 * display_size[0])
                    # box[1] = int(box[1] / 480 * display_size[1])
                    # box[2] = int(box[2] / 640 * display_size[0])
                    # box[3] = int(box[3] / 480 * display_size[1])
                    # self.get_logger().info('box: %s' % str(box))
                self.vllm_result = ''
                self.action_finish = True
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                if self.interruption:
                    msg = SetInt32.Request()
                    msg.data = 2
                    self.send_request(self.set_mode_client, msg)
                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
            # if box:
                # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2, 1)
            cv2.imshow('image', cv2.cvtColor(cv2.resize(image, (display_size[0], display_size[1])), cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1)
            if k != -1:
                break
            if not self.set_above:
                cv2.moveWindow('image', 1920 - display_size[0], 0)
                os.system("wmctrl -r image -b add,above")
                self.set_above = True
        cv2.destroyAllWindows()

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        # rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8,
                               # buffer=ros_image.data)  # Original RGB image (原始 RGB 画面)
        if self.image_queue.full():
            # If queue is full, discard the oldest image （如果队列已满，丢弃最旧的图像）
            self.image_queue.get()
            # Put image into queue （将图像放入队列）
        self.image_queue.put(rgb_image)

def main():
    node = VLLMWithCamera('vllm_with_camera')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
