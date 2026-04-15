#!/usr/bin/env python3
# encoding: utf-8
import os
import json
import uuid
import rclpy
import threading
import time
import requests
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool

from speech import speech
from large_models.config import *
from openclaw_controller.openclaw_client import OpenClawClient
from openclaw_controller.feishu_sender import FeishuSender
from openclaw_controller.config import *

from std_srvs.srv import Trigger

# 日志颜色定义
class LogColor:
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'
    RESET = '\033[0m'

class VoiceOpenClaw(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        self.openclaw = OpenClawClient()
        self.get_logger().info(f'Gateway URL: {self.openclaw.gateway_url}')

        self.declare_parameter('stream', False)
        self.openclaw.set_stream(self.get_parameter('stream').value)

        self.declare_parameter('feishu_enable', True)
        self.feishu_enable = self.get_parameter('feishu_enable').value
        self.get_logger().info(f'Feishu enable: {self.feishu_enable}')


        timer_cb_group = ReentrantCallbackGroup()

        self.wakeup_flag = False
        self.action_finish = False
        self.play_audio_finish = False
        self.running = True
        self.processing = False

        self.create_subscription(Bool, '/vocal_detect/wakeup', self.wakeup_callback, 1)

        self.create_subscription(String, '/vocal_detect/asr_result', self.asr_callback, 1)

        self.create_subscription(Bool, '/tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)

        self.create_subscription(Bool, '/voice_openclaw/feishu_enable', self.feishu_enable_callback, 1)

        self.tts_text_pub = self.create_publisher(String, '/tts_node/tts_text', 1)
        self.get_logger().info('TTS text publisher created')

        self.awake_client = self.create_client(SetBool, '/vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.get_logger().info('Wakeup service found')

        speech.play_audio(start_audio_path)

        self.get_logger().info('Start audio played')
        
        self.yolo_client = self.create_client(Trigger, '/yolo/start')

        self.feishu = FeishuSender()

        self.get_logger().info(f'{LogColor.GREEN}🦞OpenClaw node started{LogColor.RESET}')
        
        threading.Thread(target=self.process_loop, daemon=True).start()
        
    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def wakeup_callback(self, msg):
        if msg.data:
            self.get_logger().info(f'{LogColor.GREEN}Wakeup detected{LogColor.RESET}')
            self.wakeup_flag = True

    def feishu_enable_callback(self, msg):
        self.feishu_enable = msg.data
        self.get_logger().info(f'Feishu enable: {self.feishu_enable}')

    def asr_callback(self, msg):
        if msg.data and not self.processing:
            self.get_logger().info(f'{LogColor.BLUE}Openclaw Get ASR Result: {msg.data}{LogColor.RESET}')
            self.process_asr(msg.data)
            
    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data
        self.get_logger().info(f'Play finish: {msg.data}')

    def process_asr(self, text):
        if text and text.strip():
            self.processing = True
            # self.get_logger().info(f'Input: {text}')

            if self.feishu_enable:
                self.feishu.send_text(text, "user")

            try:
                reply = self.openclaw.chat(text)
                self.get_logger().info(f'{LogColor.YELLOW}Reply: {reply}{LogColor.RESET}')

                msg = String()
                msg.data = reply
                self.tts_text_pub.publish(msg)

                if self.feishu_enable:
                    self.feishu.send_text(reply, "robot")

                self.action_finish = True
            except Exception as e:
                self.get_logger().error(f'Error: {e}')
                self.processing = False
        else:
            self.get_logger().info('No speech detected')
            
    def process_loop(self):
        while self.running:
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                self.wakeup_flag = False
                self.processing = False

                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
                self.get_logger().info('TTS play finished, ready for next wakeup')
                    
            time.sleep(0.1)
        rclpy.shutdown()


def main():
    node = VoiceOpenClaw('voice_openclaw')
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
