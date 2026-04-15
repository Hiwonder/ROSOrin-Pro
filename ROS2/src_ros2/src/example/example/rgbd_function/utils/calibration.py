#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import time
import yaml
import rclpy
import queue
import threading
import numpy as np
import sdk.common as common
from rclpy.node import Node
from example.rgbd_function.utils.common import Heart
from example.rgbd_function.utils import search_plane
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from dt_apriltags import Detector
from sensor_msgs.msg import Image, CameraInfo 
from kinematics_msgs.srv import GetRobotPose
from servo_controller_msgs.msg import ServosPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position
from tf2_ros import Buffer, TransformListener, TransformException

class CalibrationNode(Node):
    hand2cam_tf_matrix = [
            [0.0, 0.0, 1.0, -0.101],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.05],
            [0.0, 0.0, 0.0, 1.0]
            ]
    
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.running = True
        self.imgpts = None
        self.imgpts1 = None
        self.plane = []
        self._init_parameters()
        self.tag_size = 0.025
        self.tag_id = [1, 2, 3]
        self.tag_id_2 = 100
        self.camera_type = os.environ['DEPTH_CAMERA_TYPE']
        # self.chassis_type = os.environ['CHASSIS_TYPE']
        self.config_file = 'transform.yaml'
        # if self.chassis_type == 'Slide_Rails':
        #     self.config_path = "/home/ubuntu/ros2_ws/src/stepper/config/"
        # else:
        self.config_path = "/home/ubuntu/ros2_ws/src/example/example/rgbd_function/config/"
        self.white_area_width = 0.167
        self.white_area_height = 0.13
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.depth_image_queue = queue.Queue(maxsize=2)
        # 创建发布者
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.result_image_pub = self.create_publisher(Image, '~/image_result', 10)
        self.finish_pub = self.create_publisher(Bool, '~/finish', 1)

        # 服务
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(Trigger, '~/start', self.start_calibration_srv_callback)

        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        
        tf_buffer = Buffer()
        self.tf_listener = TransformListener(tf_buffer, self)
        tf_future = tf_buffer.wait_for_transform_async(
            target_frame='depth_camera_link',
            source_frame='rgb_camera_link',
            time=rclpy.time.Time()
        )

        rclpy.spin_until_future_complete(self, tf_future)
        try:
            transform = tf_buffer.lookup_transform(
                'rgb_camera_link', 'depth_camera_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=5.0))
            self.static_transform = transform  # 保存变换数据
            # self.get_logger().info(f'Static transform: {self.static_transform}')
        except TransformException as e:
            self.get_logger().error(f'Failed to get static transform: {e}')

        # 提取平移和旋转
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        transform_matrix = common.xyz_quat_to_mat([translation.x, translation.y, translation.z],
                                                  [rotation.w, rotation.x, rotation.y, rotation.z])
        self.hand2cam_tf_matrix = np.matmul(transform_matrix, self.hand2cam_tf_matrix)
        self.get_logger().info(f'hand2cam_tf_matrix: {self.hand2cam_tf_matrix}')
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            self.extristric = np.array(config['extristric'])
            self.white_area_pose_cam = np.array(config['white_area_pose_cam'])
            self.white_area_pose_world = np.array(config['white_area_pose_world'])

        threading.Thread(target=self.image_processing, daemon=True).start()

    def _init_parameters(self):
        self.thread = None
        self.err_msg = None
        self.heart = None
        self.calibration_step = 0
        self.tags = []
        self.pose = []
        self.tag_count = 0
        self.K = None
        self.D = None
        self.enter = False

        self.image_sub = None
        self.camera_info_sub = None
        self.depth_image_sub = None
        self.depth_cam_info = None
        self.depth_camera_info_sub = None


    def calibration_proc(self):
        # 手眼标定。。。。(hand-eye calibration)
        # 通过固定的结构尺寸确定(calibration through fixed structural dimensions)

        # 获取当前末端坐标(get the current end-effector coordinates)
        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        self.get_logger().info(f'endpoin11111t: {endpoint}')
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation
        # self.get_logger().info(f'pose_t:{pose_t}') 
        endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
            
        # 获取标签数据(get tag data)
        t = time.time()
        self.tags = []
        self.calibration_step = 1
        while self.calibration_step == 1 and time.time() - t < 10:
            time.sleep(0.1)

        if len(self.tags) < 5:
            self.err_msg = "Time out, calibrate failed!!!"
            time.sleep(3)
            self.err_msg = None
            self.calibration_step = 0
            self.thread = None
            return

        # 识别区域中心位置标定(calibration of the center position in the recognition area)
        # 对多次识别的数据求均值(calculate the average of multiple recognition data)
        pose = map(lambda tag: common.xyz_rot_to_mat(tag.pose_t, tag.pose_R), self.tags) # 将所有位姿转为4x4齐次矩阵(convert all poses to 4x4 homogeneous matrices)
        vectors = map(lambda p: p.ravel(), pose) # 将矩阵展平为向量(flatten the matrix into a vector)
        avg_pose = np.mean(list(vectors), axis=0).reshape((4, 4))  # 求均值并重组为4x4矩阵(calculate the mean and reassemble into a 4x4 matrix)
        self.get_logger().info(f'avg_pose: {avg_pose}')
        pose_end = np.matmul(self.hand2cam_tf_matrix, avg_pose)  # 转换到末端相对坐标(transform to end-effector relative coordinates)
        self.get_logger().info(f'pose_end: {pose_end}')
        pose_world = np.matmul(endpoint, pose_end)  # 转换到机械臂世界坐标(transform to robotic arm world coordinates)
        self.get_logger().info(f'pose_world: {pose_world}')
        self.get_logger().info(f'endpoint: {endpoint}')
        self.white_area_pose_world = pose_world
        world_position = np.eye(4)
    
        world_position[:3, 3] = pose_world[:3, 3]
        
        self.white_area_pose_cam = avg_pose
        white_area_pose_cam = avg_pose.tolist()  # 识别区域中心的在相机的世界坐标系中的位置, 结果存入到param中(the position of the center of the recognition area in the camera's world coordinate system is stored in the parameter 'param')
        white_area_pose_world = world_position.tolist()  # 识别区域中心的机械臂世界坐标系的位置, 结果存入到param中(the position of the center of the recognition area in the camera's world coordinate system is stored in the parameter 'param')

        # 外参标定(extrinsic calibration)
        world_points = np.array([(-self.tag_size/2, -self.tag_size/2, 0), 
                                 ( self.tag_size/2, -self.tag_size/2, 0), 
                                 ( self.tag_size/2,  self.tag_size/2, 0), 
                                 (-self.tag_size/2,  self.tag_size/2, 0)] * len(self.tags), dtype=np.float64)

        image_points = np.array(list(map(lambda tag: tag.corners, self.tags)), dtype=np.float64).reshape((-1, 2))
        retval, rvec, tvec = cv2.solvePnP(world_points, image_points, self.K, self.D)
        rmat, _ = cv2.Rodrigues(rvec)
        tvec_flattened = tvec.flatten().tolist()

        extristric = [
            tvec_flattened,  # 将 tvec 作为第一行
            rmat[0].tolist(),  # 第一行的旋转矩阵
            rmat[1].tolist(),  # 第二行的旋转矩阵
            rmat[2].tolist()   # 第三行的旋转矩阵
        ]
        self.extristric = np.array(extristric)
        corners = self.draw_retangle()
        
        if self.camera_type == "aurora":
            data = {
                'white_area_pose_cam': white_area_pose_cam,
                'white_area_pose_world': white_area_pose_world,
                'extristric': extristric,
                'corners': corners.tolist(),
                'plane': self.plane.tolist(),
            }
        if self.camera_type == "usb_cam":
            data = {
                'white_area_pose_cam': white_area_pose_cam,
                'white_area_pose_world': white_area_pose_world,
                'extristric': extristric,
                'corners': corners.tolist(),
            }

        # self.get_logger().info(f'calibration data: {data}') 
        self.update_yaml_data(data, self.config_path + self.config_file)
        msg = Bool()
        msg.data = True
        self.finish_pub.publish(msg)  
        self.calibration_step = 20
        time.sleep(3)
        self.calibration_step = 0
        self.thread = None

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def update_yaml_data(self, new_data, yaml_file):
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)  
        else:
            data = {}  

        data.update(new_data)  

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
    
        time.sleep(0.1)

    def start_calibration_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start calibration")
        self.imgpts = None
        self.imgpts1 = None
        self.extristric = None
        self.white_area_pose_cam = None
        self.white_area_pose_world = None
        if self.image_sub is None:
            err_msg = "Please call enter service first"
            self.get_logger().info(str(err_msg))
            response.success = False
            response.message = "stop"
            return response  
        if self.thread is None:
            self.thread = threading.Thread(target=self.calibration_proc)
            self.thread.start()
            response.success = True
            response.message = "start"
            return response
        else:
            msg = "Calibration..."
            self.get_logger().info(msg)
            response.success = False
            response.message = "stop"
            return response

    def enter_srv_callback(self,request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "enter calibration")
        # 获取和发布图像的topic(get and publish topic of image)
        self._init_parameters()
        self.heart = Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)
        if self.camera_type == "aurora":
            self.image_sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
            self.camera_info_sub = self.create_subscription(CameraInfo, '/ascamera/camera_publisher/rgb0/camera_info', self.camera_info_callback, 1)
            self.depth_image_sub = self.create_subscription(Image, '/ascamera/camera_publisher/depth0/image_raw', self.depth_image_callback, 1)
            self.depth_camera_info_sub = self.create_subscription(CameraInfo, '/ascamera/camera_publisher/depth0/camera_info', self.depth_camera_info_callback, 1)
        if self.camera_type == "usb_cam":
            self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 1)
            self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)
        set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 700), (3, 86), (4, 70), (5, 500), (10, 300)))  # 设置机械臂初始位置

        self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        if self.enter:
            self.get_logger().info('\033[1;32m%s\033[0m' % "exit calibration")
            
            try:
                if self.image_sub is not None:
                    self.destroy_subscription(self.image_sub)
                    self.destroy_subscription(self.camera_info_sub)
                    self.destroy_subscription(self.depth_image_sub)
                    self.destroy_subscription(self.depth_camera_info_sub)
                    self.image_sub = None
                    self.camera_info_sub = None
                    self.depth_image_sub = None
                    self.depth_camera_info_sub = None
            except Exception as e:
                self.get_logger().error(str(e))
            self.heart.destroy()
            self.heart = None
            self.enter = False
        response.success = True
        response.message = "exit"
        return response

    def draw_retangle(self):
        white_area_center = self.white_area_pose_world.reshape(4, 4)
        white_area_cam = self.white_area_pose_cam.reshape(4, 4)
        # self.get_logger().info("white_area_center: {}".format(white_area_center))
        # self.get_logger().info("white_area_cam: {}".format(white_area_cam))
        euler_matrix = common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0))
        white_area_lt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_lb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_rb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, -self.white_area_width / 2 -0.0, 0.0), (0, 0, 0)))
        white_area_rt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, -self.white_area_width / 2 -0.0, 0.0), (0, 0, 0)))

        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        corners_cam =  np.matmul(np.linalg.inv(np.matmul(endpoint, self.hand2cam_tf_matrix)), [white_area_lt, white_area_lb, white_area_rb, white_area_rt, white_area_center])
        corners_cam = np.matmul(np.linalg.inv(white_area_cam), corners_cam)
        corners_cam = corners_cam[:, :3, 3:].reshape((-1, 3))
        tvec = self.extristric[:1]  
        rmat = self.extristric[1:]  

        while self.K is None or self.D is None:
            time.sleep(0.5)

        center_imgpts, jac = cv2.projectPoints(corners_cam[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
        self.center_imgpts = np.int32(center_imgpts).reshape(2)

        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.0)
        imgpts, jac = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)

        self.imgpts = np.int32(imgpts).reshape(-1, 2)
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03)
        imgpts, jac = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)
        self.imgpts1 = np.int32(imgpts).reshape(-1, 2)
        # self.get_logger().info('corners_cam: {}'.format(corners_cam))
        return corners_cam

    def search_plane(self, depth_image):
        p = self.depth_cam_info.p
        fx = p[0]
        fy = p[5]
        cx = p[2]
        cy = p[6]
        height = self.depth_cam_info.height
        width = self.depth_cam_info.width
        camera_intrinsics = [fx,fy,cx,cy]
        # self.get_logger().info(f'{width}, {height} {camera_intrinsics}')
        searcher = search_plane.SearchPlane(width, height, camera_intrinsics)
        a, m, s = searcher.find_plane(depth_image)
        self.plane = m

    def image_processing(self):
        while self.running:
            if self.enter:
                rgb_image = self.image_queue.get(block=True)
                if self.camera_type == 'aurora':
                    depth_image = self.depth_image_queue.get(block=True)
                result_image = np.copy(rgb_image)
                if self.K is not None:
                    tags = self.at_detector.detect(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), True, (self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]), self.tag_size)
                    result_image = common.draw_tags(result_image, tags)
                    
                    if self.calibration_step == 1:
                        if len(tags) == 1 and (tags[0].tag_id in self.tag_id or tags[0].tag_id == self.tag_id_2):
                            self.err_msg = None
                            if len(self.tags) > 0:
                                if common.distance(self.tags[-1].pose_t, tags[0].pose_t) < 0.003:
                                    self.tags.append(tags[0])
                                else:
                                    self.tags = []
                            else:
                                self.tags.append(tags[0])
                            if len(self.tags) >= 10:
                                print("收集完成")
                                if self.camera_type == 'aurora':
                                    if self.depth_cam_info.k is not None:
                                        self.search_plane(depth_image)
                                self.calibration_step = 2
                        else:
                            self.tags = []
                            if self.err_msg is None:
                                self.err_msg = "Please make sure there is only one tag in the;screen and the tag id is 1 or 100"

                    if self.extristric is not None:
                        # 添加保护，避免 extristric 或 K 为 None
                        if self.extristric is None or self.K is None:
                            return        # 等数据齐全再做投影
                        
                        world_points = np.array([(-self.tag_size/2, -self.tag_size/2, 0),
                                                ( self.tag_size/2, -self.tag_size/2, 0),
                                                ( self.tag_size/2,  self.tag_size/2, 0),
                                                (-self.tag_size/2,  self.tag_size/2, 0)], dtype=np.float64)
                        
                        # ------- 取旋转、平移 ---------
                        rmat = self.extristric[1:].astype(np.float64).reshape(3, 3)   # 3×3
                        tvec = self.extristric[0].astype(np.float64).reshape(3, 1)    # 3×1

                        # ------- 投影 ---------
                        image_points, _ = cv2.projectPoints(
                                world_points,
                                rmat,
                                tvec,
                                self.K,
                                self.D
                        )

                        image_points = image_points.astype(np.int32).reshape((-1, 2)).tolist()
                        for p in image_points:
                            cv2.circle(result_image, tuple(p), 3, (0, 0, 0), -1)

                if self.err_msg is not None:
                    self.get_logger().info(str(self.err_msg))
                    err_msg = self.err_msg.split(';')
                    for i, m in enumerate(err_msg):
                        cv2.putText(result_image, m, (5, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 6)
                        cv2.putText(result_image, m, (5, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if self.calibration_step != 0:
                    if self.calibration_step == 20:
                        msg = "Calibration finished!"
                        self.draw_retangle()
                        cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
                    else:
                        msg = "Calibrating..."
                    cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 6)
                    cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if self.imgpts1 is not None:
                    cv2.drawContours(result_image, [self.imgpts1], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制轮廓(draw contours)
                if self.imgpts is not None:
                    cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
                # 发布结果图像( publish the resulting image)
                self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))
            else:
                time.sleep(0.1)

    def camera_info_callback(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64)

    def depth_camera_info_callback(self, msg):
        self.depth_cam_info = msg

    def image_callback(self, ros_image):
        # 将ros格式图像转换为opencv格式(convert the ros format image to opencv format)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)

        if self.image_queue.full():
            # # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # # 将图像放入队列
        self.image_queue.put(rgb_image)

    def depth_image_callback(self, ros_depth_image):
        depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                 buffer=ros_depth_image.data)

        if self.depth_image_queue.full():
            # # 如果队列已满，丢弃最旧的图像
            self.depth_image_queue.get()
        # # 将图像放入队列
        self.depth_image_queue.put(depth_image)

def main():
    node = CalibrationNode('calibration')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.running = False  # 停止线程标志
        executor.shutdown()
 
if __name__ == "__main__":
    main()
