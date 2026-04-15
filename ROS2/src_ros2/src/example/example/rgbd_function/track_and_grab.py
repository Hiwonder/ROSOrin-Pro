#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/12/19
# 机械前超前看识别追踪空中指定颜色物品(Mechanical front-facing vision for recognizing and tracking aerial objects of a specific color)
# 通过深度相机识别计算物品的空间位置(Determine the spatial location of objects using a depth camera)
# 完成抓取并放到指定位置(Complete the grasping and place the object at the designated location)
import cv2
import math
import time
import rclpy
import queue
import signal
import threading
import numpy as np
import message_filters
from rclpy.node import Node
from std_srvs.srv import SetBool
from sdk import pid, common, fps
from std_srvs.srv import Trigger
from interfaces.srv import SetString
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller.bus_servo_control import set_servo_position


def depth_pixel_to_camera(pixel_coords, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    px, py = pixel_coords
    x = (px - cx) * depth / fx
    y = (py - cy) * depth / fy
    z = depth
    return np.array([x, y, z])

class ColorTracker:
    def __init__(self, target_color):
        self.target_color = target_color
        self.pid_yaw = pid.PID(20.5, 1.0, 1.2)
        self.pid_pitch = pid.PID(20.5, 1.0, 1.2)
        self.yaw = 500
        self.pitch = 150
    
    def proc(self, source_image, result_image, color_ranges):
        h, w = source_image.shape[:2]
        color = color_ranges['lab']['Stereo'][self.target_color]

        img = cv2.resize(source_image, (int(w/2), int(h/2)))
        img_blur = cv2.GaussianBlur(img, (3, 3), 3) # 高斯模糊(Gaussian blur)
        img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB) # 转换到 LAB 空间(convert to the LAB space)
        mask = cv2.inRange(img_lab, tuple(color['min']), tuple(color['max'])) # 二值化(binarization)

        # 平滑边缘，去除小块，合并靠近的块(Smooth edges, remove small patches, and merge nearby blocks)
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # 找出最大轮廓(Find the largest contour)
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        min_c = None
        for c in contours:
            if math.fabs(cv2.contourArea(c)) < 50:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(c) # 最小外接圆(Minimum enclosing circle)
            if min_c is None:
                min_c = (c, center_x)
            elif center_x < min_c[1]:
                # if center_x < min_c[1]:
                    min_c = (c, center_x)

        # 如果有符合要求的轮廓(if there is a contour that meets the requirements)
        if min_c is not None:
            (center_x, center_y), radius = cv2.minEnclosingCircle(min_c[0]) # 最小外接圆(minimum enclosing circle)

            # 圈出识别的的要追踪的色块(Circle the color blocks to be tracked)
            circle_color = common.range_rgb[self.target_color] if self.target_color in common.range_rgb else (0x55, 0x55, 0x55)
            cv2.circle(result_image, (int(center_x * 2), int(center_y * 2)), int(radius * 2), circle_color, 2)

            center_x = center_x * 2
            center_x_1 = center_x / w
            if abs(center_x_1 - 0.5) > 0.02: # 相差范围小于一定值就不用再动了(If the difference range is less than a certain value, no further adjustments are needed)
                self.pid_yaw.SetPoint = 0.5 # 我们的目标是要让色块在画面的中心, 就是整个画面的像素宽度的 1/2 位置(Our goal is to center the color blocks in the middle of the frame, which is at the 1/2 position of the frame's pixel width.)
                self.pid_yaw.update(center_x_1)
                self.yaw = min(max(self.yaw + self.pid_yaw.output, 0), 1000)
            else:
                self.pid_yaw.clear() # 如果已经到达中心了就复位一下 pid 控制器(If the center is reached, reset the PID controller)

            center_y = center_y * 2
            center_y_1 = center_y / h
            if abs(center_y_1 - 0.5) > 0.02:
                self.pid_pitch.SetPoint = 0.5
                self.pid_pitch.update(center_y_1)
                self.pitch = min(max(self.pitch + self.pid_pitch.output, 100), 720)
            else:
                self.pid_pitch.clear()
            return (result_image, (self.pitch, self.yaw), (center_x, center_y), radius * 2)
        else:
            return (result_image, None, None, 0)


class TrackAndGrabNode(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101], # -0.101
    [-1.0, 0.0, 0.0, 0.0], # 0.011
    [0.0, -1.0, 0.0, 0.037], # 0.045
    [0.0, 0.0, 0.0, 1.0]
]

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.fps = fps.FPS()
        self.moving = False
        self.count = 0
        self.start = False
        self.running = True
        self.last_pitch_yaw = (0, 0)

        self.enable_disp = 1
        signal.signal(signal.SIGINT, self.shutdown)
        self.lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
        self.last_position = (0, 0, 0)
        self.stamp = time.time()

        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)

        self.target_color = None
     
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose')
        self.get_current_pose_client.wait_for_service()
        self.set_pose_target_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.set_pose_target_client.wait_for_service()

        self.create_service(Trigger, '~/start', self.start_srv_callback)
        self.create_service(Trigger, '~/stop', self.stop_srv_callback)
        self.create_service(SetString, '~/set_color', self.set_color_srv_callback)
        self.tracker = None

        self.image_queue = queue.Queue(maxsize=2)
        self.endpoint = None

        self.start_stamp = time.time() + 3

        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        # self.client = self.create_client(SetBool, '/depth_cam/set_ldp_enable')
        # self.client.wait_for_service()

        rgb_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb0/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth0/image_raw')
        info_sub = message_filters.Subscriber(self, CameraInfo, '/depth_cam/depth0/camera_info')

        # 同步时间戳, 时间允许有误差在0.03s(Synchronize the timestamps, with an allowable time error of 0.03 seconds)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, info_sub], 3, 0.02)
        sync.registerCallback(self.multi_callback) #执行反馈函数(Execute the feedback function)
        
        timer_cb_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        # msg = SetBool.Request()
        # msg.data = False
        # self.send_request(self.client, msg)

        # set_servo_position(self.joints_pub, 1, ((1, 500), (2, 720), (3, 100), (4, 120), (5, 500), (10, 600)))
        set_servo_position(self.joints_pub, 1, ((1, 500), (2, 700), (3, 151), (4, 70), (5, 500), (10, 150)))
        time.sleep(1)
        if self.get_parameter('start').value:
            self.target_color = self.get_parameter('color').value
 
            msg = SetString.Request()
            msg.data = self.target_color
            self.set_color_srv_callback(msg, SetString.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def shutdown(self, signum, frame):
        self.running = False
        self.get_logger().info('\033[1;32m%s\033[0m' % "shutdown")
        # 确保程序能够正常退出
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
        rclpy.shutdown()  # 关闭ROS2
        import sys
        sys.exit(0)  # 强制退出程序

    def set_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_color")
        self.target_color = request.data
        self.tracker = ColorTracker(self.target_color)
        self.get_logger().info('\033[1;32mset color: %s\033[0m' % self.target_color)
        self.start = True
        response.success = True
        response.message = "set_color"
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop")
        self.start = False
        self.moving = False
        self.count = 0
        self.last_pitch_yaw = (0, 0)
        self.last_position = (0, 0, 0)
        set_servo_position(self.joints_pub, 1, ((1, 500), (2, 720), (3, 100), (4, 120), (5, 500), (10, 150)))
        response.success = True
        response.message = "stop"
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def multi_callback(self, ros_rgb_image, ros_depth_image, depth_camera_info):
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像(If the queue is full, discard the oldest image)
            self.image_queue.get()
        # 将图像放入队列(Add the image into the queue)
        self.image_queue.put((ros_rgb_image, ros_depth_image, depth_camera_info))

    def get_endpoint(self):
        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request()).pose
        self.endpoint = common.xyz_quat_to_mat([endpoint.position.x, endpoint.position.y, endpoint.position.z],
                                        [endpoint.orientation.w, endpoint.orientation.x, endpoint.orientation.y, endpoint.orientation.z])
        return self.endpoint

    def pick(self, position):
        if position[2] < 0.2:
            yaw = 80
        else:
            yaw = 30
        msg = set_pose_target(position, yaw, [-180.0, 180.0], 1.0)
        res = self.send_request(self.set_pose_target_client, msg)
        if res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1, ((1, servo_data[0]), ))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.5, ((1, servo_data[0]),(2, servo_data[1]), (3, servo_data[2]),(4, servo_data[3]), (5, servo_data[4])))
            time.sleep(1.5)
        set_servo_position(self.joints_pub, 0.5, ((10, 500),))
        time.sleep(1)
        position[2] += 0.03

        msg = set_pose_target(position, yaw, [-180.0, 180.0], 1.0)
        res = self.send_request(self.set_pose_target_client, msg)
        if res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1, ((1, servo_data[0]),(2, servo_data[1]), (3, servo_data[2]),(4, servo_data[3]), (5, servo_data[4])))
            time.sleep(1)
        set_servo_position(self.joints_pub, 1, ((1, 500), (2, 720), (3, 100), (4, 120), (5, 500), (10, 500)))
        time.sleep(1)
        set_servo_position(self.joints_pub, 1, ((1, 125), (2, 635), (3, 120), (4, 200), (5, 500)))
        time.sleep(1)
        set_servo_position(self.joints_pub, 1.5, ((1, 125), (2, 325), (3, 267), (4, 290), (5, 500)))
        time.sleep(1.5)
        set_servo_position(self.joints_pub, 1, ((1, 125), (2, 325), (3, 267), (4, 290), (5, 500), (10, 150)))
        time.sleep(1.5)
        set_servo_position(self.joints_pub, 1, ((1, 500), (2, 700), (3, 150), (4, 70), (5, 500), (10, 150)))
        time.sleep(2)
        self.tracker.yaw = 500
        self.tracker.pitch = 150
        self.tracker.pid_yaw.clear()
        self.tracker.pid_pitch.clear()
        self.stamp = time.time()
        self.moving = False

    def main(self):
        while self.running:
            try:
                ros_rgb_image, ros_depth_image, depth_camera_info = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue
            try:
                rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8, buffer=ros_rgb_image.data)
                depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16, buffer=ros_depth_image.data)
                result_image = np.copy(rgb_image)

                h, w = depth_image.shape[:2]
                depth = np.copy(depth_image).reshape((-1, ))
                depth[depth<=0] = 55555

                sim_depth_image = np.clip(depth_image, 0, 2000).astype(np.float64)

                sim_depth_image = sim_depth_image / 2000.0 * 255.0
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)

                if self.tracker is not None and self.moving == False and time.time() > self.start_stamp and self.start:
                    result_image, p_y, center, r = self.tracker.proc(rgb_image, result_image, self.lab_data)
                    if p_y is not None:
                        set_servo_position(self.joints_pub, 0.02, ((1, int(p_y[1])), (4, int(p_y[0]))))
                        center_x, center_y = center
                        if center_x > w:
                            center_x = w
                        if center_y > h:
                            center_y = h
                        if abs(self.last_pitch_yaw[0] - p_y[0]) < 3 and abs(self.last_pitch_yaw[1] - p_y[1]) < 3:
                            if time.time() - self.stamp > 2:
                                self.stamp = time.time()
                                roi = [int(center_y) - 5, int(center_y) + 5, int(center_x) - 5, int(center_x) + 5]
                                if roi[0] < 0:
                                    roi[0] = 0
                                if roi[1] > h:
                                    roi[1] = h
                                if roi[2] < 0:
                                    roi[2] = 0
                                if roi[3] > w:
                                    roi[3] = w
                                roi_distance = depth_image[roi[0]:roi[1], roi[2]:roi[3]]
                                try:
                                    dist = round(float(np.mean(roi_distance[np.logical_and(roi_distance>0, roi_distance<10000)])/1000.0), 3)
                                except BaseException as e:
                                    self.get_logger().info('error2: ' + str(e))
                                    txt = "DISTANCE ERROR !!!"
                                    return
                                if np.isnan(dist):
                                    txt = "DISTANCE ERROR !!!"
                                    return
                                dist += 0.015 # 物体半径补偿(Object radius compensation)
                                dist += 0.015 # 误差补偿(Error compensation)
                                K = depth_camera_info.k
                                self.get_endpoint()
                                position = depth_pixel_to_camera((center_x, center_y), dist, (K[0], K[4], K[2], K[5]))
                                
                                position[0] -= 0.01  # rgb相机和深度相机tf有1cm偏移(There is a 1 cm offset between the RGB camera and the depth camera in the TF.)
                                position[1] -= 0.02
                                position[2] -= 0.01
								
                                pose_end = np.matmul(self.hand2cam_tf_matrix, common.xyz_euler_to_mat(position, (0, 0, 0)))  # 转换的末端相对坐标(Relative coordinates of the end effector after transformation)
                                world_pose = np.matmul(self.endpoint, pose_end)  # 转换到机械臂世界坐标(Transform to the robot arm's world coordinates)
                                pose_t, pose_R = common.mat_to_xyz_euler(world_pose)
                                self.stamp = time.time()
                                self.moving = True
                                threading.Thread(target=self.pick, args=(pose_t,)).start()
                        else:
                            self.stamp = time.time()
                        dist = depth_image[int(center_y),int(center_x)]
                        if dist < 100:
                            txt = "TOO CLOSE !!!"
                        else:
                            txt = "Dist: {}mm".format(dist)
                        cv2.circle(result_image, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)
                        cv2.circle(depth_color_map, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)
                        cv2.putText(depth_color_map, txt, (10, 400 - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 10, cv2.LINE_AA)
                        cv2.putText(depth_color_map, txt, (10, 400 - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
                        self.last_pitch_yaw = p_y
                    else:
                        self.stamp = time.time()
                if self.enable_disp:
                    # result_image = np.concatenate([cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), depth_color_map, ], axis=1)
                    result_image = np.concatenate([result_image, depth_color_map], axis=1)
                    
                    cv2.imshow("depth", result_image)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27:  # 按q或者esc退出(Hold q or esc key to exit)
                        self.running = False

            except Exception as e:
                self.get_logger().info('error1: ' + str(e))
        rclpy.shutdown()

def main():
    node = TrackAndGrabNode('track_and_grab')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
