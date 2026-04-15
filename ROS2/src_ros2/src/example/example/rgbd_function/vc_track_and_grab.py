#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/11/18
# @author:aiden
# 语音控制颜色追踪(voice control for color tracking)
import os
import json
import rclpy
import threading
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from interfaces.srv import SetString
from xf_mic_asr_offline import voice_play
from rclpy.executors import MultiThreadedExecutor
from ros_robot_controller_msgs.msg import BuzzerState
from rclpy.callback_groups import ReentrantCallbackGroup

class VoiceControlGrabNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.language = os.environ['ASR_LANGUAGE']
        self.create_subscription(String, '/asr_node/voice_words', self.words_callback, 1)
        self.client = self.create_client(Trigger, '/asr_node/init_finish')
        self.client.wait_for_service()
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/track_and_grab/init_finish')
        self.client.wait_for_service()
        self.set_color_client = self.create_client(SetString, '/track_and_grab/set_color', callback_group=timer_cb_group)
        self.set_color_client.wait_for_service()
        self.stop_client = self.create_client(Trigger, '/track_and_grab/stop', callback_group=timer_cb_group)
        self.stop_client.wait_for_service()
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 1)
        self.play('running')
        self.get_logger().info('唤醒口令: 小幻小幻(Wake up word: hello hiwonder)')
        self.get_logger().info('唤醒后15秒内可以不用再唤醒(No need to wake up within 15 seconds after waking up)')
        self.get_logger().info('控制指令: 追踪红色 追踪绿色 追踪蓝色 停止追踪(Voice command: track red/green/blue)')
        
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def play(self, name):
        voice_play.play(name, language=self.language)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def words_callback(self, msg):
        words = json.dumps(msg.data, ensure_ascii=False)[1:-1]
        if self.language == 'Chinese':
            words = words.replace(' ', '')
        self.get_logger().info('words: %s'%words)
        if words is not None and words not in ['唤醒成功(wake-up-success)', '休眠(Sleep)', '失败5次(Fail-5-times)',
                                               '失败10次(Fail-10-times']:
            if words == '追踪红色' or words == 'track red':
                msg = SetString.Request()
                msg.data = "red"
                res = self.send_request(self.set_color_client, msg)
                if res.success:
                    self.play('ok')
                else:
                    self.play('open_fail')
            elif words == '追踪绿色' or words == 'track green':
                msg = SetString.Request()
                msg.data = "green"
                res = self.send_request(self.set_color_client, msg)
                if res.success:
                    self.play('ok')
                else:
                    self.play('open_fail')
            elif words == '追踪蓝色' or words == 'track blue':
                msg = SetString.Request()
                msg.data = "blue"
                res = self.send_request(self.set_color_client, msg)
                if res.success:
                    self.play('ok')
                else:
                    self.play('open_fail')
            elif words == '停止追踪' or words == 'stop tracking':
                msg = Trigger.Request()
                res = self.send_request(self.stop_client, msg)
                if res.success:
                    self.play('stop')
                else:
                    self.play('stop_fail')
        elif words == '唤醒成功(wake-up-success)':
            self.play('awake')
        elif words == '休眠(Sleep)':
            msg = BuzzerState()
            msg.freq = 1900
            msg.on_time = 0.05
            msg.off_time = 0.01
            msg.repeat = 1
            self.buzzer_pub.publish(msg)

def main():
    node = VoiceControlGrabNode('vc_track_and_grab')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()

