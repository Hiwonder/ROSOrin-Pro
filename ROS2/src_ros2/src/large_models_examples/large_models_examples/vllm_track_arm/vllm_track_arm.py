#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import cv2
import json
import time
import queue
import rclpy
import threading
import PIL.Image
import numpy as np
import sdk.fps as fps
import message_filters
from sdk import common
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Bool
from std_srvs.srv import Trigger, SetBool, Empty
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from speech import speech
from large_models.config import *
from large_models_msgs.srv import SetString, SetModel, SetInt32
from large_models_examples.vllm_track_arm.arm_track_anything import ObjectTracker
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from large_models_examples.tracker import Tracker
import pycuda.driver as cuda
cuda.init()  # 确保CUDA已经初始化

# (import_module) 导入机械臂控制相关模块
import signal
import sdk.pid as pid
from kinematics import transform
from kinematics_msgs.srv import SetRobotPose
from interfaces.msg import ColorsInfo, ColorDetect
from interfaces.srv import SetColorDetectParam, SetString as SetStringSrv
from kinematics.kinematics_control import set_pose_target

language = os.environ["ASR_LANGUAGE"]
if language == 'Chinese':
    PROMPT = '''
你作为智能车，善于图像识别，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出，然后进行跟随。
## 1. 理解用户指令
我会给你一句话，你需要根据我的话中提取「物体名称」。 **object对应的name要用英文表示**, **不要输出没有提及到的物体**
## 2. 理解图片
我会给你一张图, 从这张图中找到「物体名称」对应物体的左上角和右下角的像素坐标; 如果没有找到，那xyxy为[]。**不要输出没有提及到的物体**
【特别注意】： 要深刻理解物体的方位关系, response需要结合用户指令和检测的结果进行回答
## 输出格式（请仅输出以下内容，不要说任何多余的话)
{
    "object": "name", 
    "xyxy": [xmin, ymin, xmax, ymax],
    "response": "5到30字的中文回答"
}
    '''
else:
    PROMPT = '''
**Role
You are a smart car with advanced visual recognition capabilities. Your task is to analyze an image sent by the user, perform object detection, and follow the detected object. Finally, return the result strictly following the specified output format.

Step 1: Understand User Instructions
You will receive a sentence. From this sentence, extract the object name to be detected.
Note: Use English for the object value, do not include any objects not explicitly mentioned in the instruction.

Step 2: Understand the Image
You will also receive an image. Locate the target object in the image and return its coordinates as the top-left and bottom-right pixel positions in the form [xmin, ymin, xmax, ymax].
Note: If the object is not found, then "xyxy" should be an empty list: [], only detect and report objects mentioned in the user instruction.The coordinates (xmin, ymin, xmax, ymax) must be normalized to the range [0, 1]

**Important: Accurately understand the spatial position of the object. The "response" must reflect both the user's instruction and the detection result.

**Output Format (strictly follow this format, do not output anything else.The coordinates (xmin, ymin, xmax, ymax) must be normalized to the range [0, 1])
{
    "object": "name", 
    "xyxy": [xmin, ymin, xmax, ymax],
    "response": "reflect both the user's instruction and the detection result (5-30 characters)"
}

**Example
Input: track the person
Output:
{
    "object": "person",
    "xyxy": [0.1, 0.3, 0.4, 0.6],
    "response": "I have detected a person in a white T-shirt and will track him now."
}
    '''

display_size = [int(640*6/4), int(480*6/4)]


class Center:
    def __init__(self, x, y, width, height, radius):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.radius = radius


class VLLMArmTrack(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        # initialization arm control parameters(初始化机械臂控制参数)
        self.z_dis = 0.22
        self.y_dis = 500
        self.x_init = transform.link3 + transform.tool_link
        self.center_x = None
        self.center_y = None
        self.running = True
        self.arm_start = False

        # initialization PID controller(初始化PID控制器)
        self.pid_z = pid.PID(0.00004, 0.0, 0.0)
        self.pid_y = pid.PID(0.05, 0.0, 0.0)

        signal.signal(signal.SIGINT, self.shutdown)
        self.camera_type = os.environ.get('DEPTH_CAMERA_TYPE')

        # large model tracking initialization(大模型追踪相关初始化)
        self.fps = fps.FPS() # Frame rate counter(帧率统计器)
        self.image_queue = queue.Queue(maxsize=2)
        self.vllm_result = ''
        self.set_above = False
        self.data = []
        self.box = []
        self.stop = True
        self.start_track = False
        self.action_finish = False
        self.play_audio_finish = False
        self.track = ObjectTracker(use_mouse=True, automatic=True, log=self.get_logger())

        timer_cb_group = ReentrantCallbackGroup()
        self.client = speech.OpenAIAPI(api_key, base_url)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)

        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.create_subscription(String, 'agent_process/result', self.vllm_result_callback, 1)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1)

        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.set_mode_client = self.create_client(SetInt32, 'vocal_detect/set_mode')
        self.set_mode_client.wait_for_service()
        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()

        self.camera = 'usb_cam'
        if 'usb_cam' in self.camera_type:
            self.camera = 'depth_cam'
        else:
            self.camera = 'depth_cam'
        self.image_sub = self.create_subscription(Image, '/%s/rgb0/image_raw' % self.camera, self.image_callback, 1)  # Subscribe to the camera(摄像头订阅)
        self.bridge = CvBridge()

        # arm_control(机械臂控制服务)
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()

        # create service to start/stop arm tracking (创建服务用于启动/停止机械臂追踪)
        self.create_service(Trigger, '~/start_arm_track', self.start_arm_track_callback)
        self.create_service(Trigger, '~/stop_arm_track', self.stop_arm_track_callback, callback_group=timer_cb_group)

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def shutdown(self, signum, frame):
        self.running = False

    def init_process(self):
        self.timer.cancel()
        msg = SetModel.Request()
        msg.model_type = 'vllm'
        if language == 'Chinese':
            msg.model = stepfun_vllm_model
            msg.api_key = stepfun_api_key
            msg.base_url = stepfun_base_url
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = PROMPT
        self.send_request(self.set_prompt_client, msg)

        self.init_arm_action()

        self.mecanum_pub.publish(Twist())
        time.sleep(1.8)
        speech.play_audio(start_audio_path)
        threading.Thread(target=self.process, daemon=True).start()
        threading.Thread(target=self.display_thread, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        self.get_logger().info('\033[1;32m%s\033[0m' % PROMPT)

    def init_arm_action(self):
        msg = set_pose_target([self.x_init, 0.0, self.z_dis], 0.0, [-180.0, 180.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1.5, ((10, 500), (5, 500), (4, servo_data[3]), (3, servo_data[2]), (2, servo_data[1]), (1, servo_data[0])))
            time.sleep(1.8)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def start_arm_track_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start arm track")
        self.arm_start = True
        response.success = True
        response.message = "start arm track"
        return response

    def stop_arm_track_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop arm track")
        self.arm_start = False
        response.success = True
        response.message = "stop arm track"
        return response

    def wakeup_callback(self, msg):
        if msg.data and self.vllm_result:
            self.get_logger().info('wakeup interrupt')
            self.track.stop()
            self.stop = True
            self.arm_start = False
        elif msg.data and not self.stop:
            self.get_logger().info('wakeup interrupt')
            self.track.stop()
            self.stop = True
            self.arm_start = False

    def vllm_result_callback(self, msg):
        self.vllm_result = msg.data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def process(self):
        box = ''
        while self.running:
            if self.vllm_result:
                try:
                    # self.get_logger().info('vllm_result: %s' % self.vllm_result)
                    if self.vllm_result.startswith("```") and self.vllm_result.endswith("```"):
                        self.vllm_result = self.vllm_result.strip("```").replace("json\n", "").strip()
                    self.vllm_result = json.loads(self.vllm_result)
                    response = self.vllm_result['response']
                    msg = String()
                    msg.data = response
                    self.tts_text_pub.publish(msg)
                    box = self.vllm_result['xyxy']
                    if box:
                        if language == 'Chinese':
                            box = self.client.data_process(box, 640, 480)
                            self.get_logger().info('box: %s' % str(box))
                        else:
                            box = [int(box[0] * 640), int(box[1] * 480), int(box[2] * 640), int(box[3] * 480)]
                        # self.get_logger().info('box: %s' % str(box))
                        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        box[0] = int(box[0] / 640 * display_size[0])
                        box[1] = int(box[1] / 480 * display_size[1])
                        box[2] = int(box[2] / 640 * display_size[0])
                        box[3] = int(box[3] / 480 * display_size[1])
                        self.get_logger().info('box: %s' % str(box))
                        self.box = box
                except (ValueError, TypeError):
                    self.box = []
                    msg = String()
                    msg.data = self.vllm_result
                    self.tts_text_pub.publish(msg)
                self.vllm_result = ''
                self.action_finish = True
            else:
                time.sleep(0.02)
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
                msg = SetInt32.Request()
                msg.data = 1
                self.send_request(self.set_mode_client, msg)
                self.stop = False


    def display_thread(self):
        # Create a new CUDA context in the current thread(在当前线程创建一个新的 CUDA 上下文)
        dev = cuda.Device(0)
        ctx = dev.make_context()
        try:
            model_path = os.path.split(os.path.realpath(__file__))[0]
            back_exam_engine_path = os.path.join(model_path, "../resources/models/nanotrack_backbone_exam.engine")
            back_temp_engine_path = os.path.join(model_path, "../resources/models/nanotrack_backbone_temp.engine")
            head_engine_path = os.path.join(model_path, "../resources/models/nanotrack_head.engine")
            tracker = Tracker(back_exam_engine_path, back_temp_engine_path, head_engine_path)
            while self.running:
                image = self.image_queue.get(block=True)
                image = cv2.resize(image, tuple(display_size))
                h,w =  image.shape[:2]
                if self.box:
                    self.track.set_track_target(tracker, self.box, image)
                    self.start_track = True
                    self.box = []
                if self.start_track:
                    self.data = self.track.track_pixel(tracker, image)
                    image = self.data[-1]
                    self.center_x = self.data[0]
                    self.center_y = self.data[1]
                    self.arm_start = True
                if self.arm_start:
                    if self.center_x is not None  and self.center_y is not None and self.arm_start:
                        t1 = time.time()

                        self.pid_y.SetPoint = w / 2
                        self.pid_y.update(self.center_x)
                        self.y_dis += self.pid_y.output
                        if self.y_dis < 200:
                            self.y_dis = 200
                        if self.y_dis > 800:
                            self.y_dis = 800

                        self.pid_z.SetPoint = h / 2 
                        self.pid_z.update(self.center_y)
                        self.z_dis += self.pid_z.output
                        if self.z_dis > 0.32:
                            self.z_dis = 0.32
                        if self.z_dis < 0.18:
                            self.z_dis = 0.18
                        # set the target position of arm (设置机械臂目标位置)
                        msg = set_pose_target([self.x_init, 0.0, self.z_dis], 0.0, [-180.0, 180.0], 1.0)
                        res = self.send_request(self.kinematics_client, msg)
                        t2 = time.time()
                        t = t2 - t1
                        if t < 0.02:
                            time.sleep(0.02 - t)
                        if res.pulse:
                            servo_data = res.pulse
                            set_servo_position(self.joints_pub, 0.02, ((10, 500), (5, 500), (4, servo_data[3]), (3, servo_data[2]), (2, servo_data[1]), (1, int(self.y_dis))))
                        else:
                            set_servo_position(self.joints_pub, 0.02, ((1, int(self.y_dis)), ))
                    else:
                        time.sleep(0.01)
                self.fps.update()
                self.fps.show_fps(image)
                cv2.imshow('image', image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # Press Q or Esc to quit(按q或者esc退出)
                    self.mecanum_pub.publish(Twist())
                    self.running = False
                if not self.set_above:
                    cv2.moveWindow('image', 1920 - display_size[0], 0)
                    os.system("wmctrl -r image -b add,above")
                    self.set_above = True

            cv2.destroyAllWindows()
        finally:
            # Ensure the context is properly released(确保上下文被正确释放)
            ctx.pop()

    def image_callback(self,ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # If the queue is full, discard the oldest image(如果队列已满，丢弃最旧的图像)
            self.image_queue.get()
        # Put the image into the queue(将图像放入队列)
        self.image_queue.put(bgr_image)

    def get_node_state(self, request, response):
        return response

def main():
    node = VLLMArmTrack('vllm_arm_track')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()