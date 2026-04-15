#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/03/24
# @author:aiden
# ar增强(augmented reality)
import os
import cv2
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from app.common import Heart
from apriltag import apriltag
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from interfaces.srv import SetString
from app.obj_loader import OBJ as obj_load
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

# Acquire the default storage path for the model.(获取模型默认存放路径)
MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0])), 'ar_models')

# Solve for the points of pnp, the four corners and the center point of the square.(求解pnp的点，正方形的四个角和中心点)
OBJP = np.array([[-1, -1,  0],
                 [ 1, -1,  0],
                 [-1,  1,  0],
                 [ 1,  1,  0],
                 [ 0,  0,  0]], dtype=np.float32)

# Draw the coordinate of the cube(绘制立方体的坐标)
AXIS = np.float32([[-1, -1, 0], 
                   [-1,  1, 0], 
                   [ 1,  1, 0], 
                   [ 1, -1, 0],
                   [-1, -1, 2],
                   [-1,  1, 2],
                   [ 1,  1, 2],
                   [ 1, -1, 2]])

# Model scaling(模型缩放比例)
MODELS_SCALE = {
                'bicycle': 50, 
                'fox': 4, 
                'chair': 400, 
                'cow': 0.4,
                'wolf': 0.6,
                }

def draw_rectangle(img, imgpts):
    '''
    Draw the cube(绘制立方体)
    :param img: The image to draw the cube(要绘制立方体的图像)
    :param imgpts: Angular point of the cube(立方体的角点)
    :return: The image to draw the cube(要绘制立方体的图像)
    '''
    imgpts = np.int32(imgpts).reshape(-1, 2)
    cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)  # Draw contour points, filled.(绘制轮廓点，填充形式)
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)  # Draw points connected by lines.(绘制线连接点)
    cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)  # Draw contour points, unfilled.(绘制轮廓点，不填充)
    
    return img

class ARNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        # Camera Intrinsic Parameters（摄像头内参）
        self.camera_intrinsic = np.matrix([[619.063979, 0,          302.560920],
                                           [0,          613.745352, 237.714934],
                                           [0,          0,          1]])
        self.dist_coeffs = np.array([0.103085, -0.175586, -0.001190, -0.007046, 0.000000])
        
        self.obj = None
        self.image_sub = None
        self.target_model = None
        self.camera_info_sub = None
        self.bridge = CvBridge()
        self.machine = os.environ['MACHINE_TYPE']
        self.camera = os.environ['DEPTH_CAMERA_TYPE']
        self.tag_detector = apriltag("tag36h11")  # Instantiate apriltag(实例化apriltag)
        self.lock = threading.RLock()  # Thread lock(线程锁)
        
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)  # Publish the final image(发布最终图像)
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)  # Enter the game service(进入发玩法服务)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)  # Exit the game service(退出玩法服务)
        Heart(self, self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # Heartbeat package(心跳包)
        self.create_service(SetString, '~/set_model', self.set_model_srv_callback)  # Set the model service.(设置模型服务)
        self.debug = self.get_parameter('debug').value
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        # 进入玩法(Enter)
        if self.debug:
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())

    def get_node_state(self, request, response):
        response.success = True
        return response

    def enter_srv_callback(self, request, response):
        # Enter the service(进入服务)
        self.get_logger().info('\033[1;32m%s\033[0m' % "ar enter")
        # If there is a node when entering the service, cancel subscription and subscribe again.(进入服务时如果节点还在则注销订阅，重新订阅)
         
        with self.lock:
            self.obj = None
            self.target_model = None
            if self.image_sub is None:
                self.image_sub = self.create_subscription(Image, '/depth_cam/rgb0/image_raw', self.image_callback, 1)  # Subscribe to the camera(摄像头订阅)

            if self.camera_info_sub is None:
                self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/rgb0/camera_info', self.camera_info_callback, 1) # Subscribe to the camera information(订阅摄像头信息)
        
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        # Exit the service(退出服务)
        self.get_logger().info('\033[1;32m%s\033[0m' % "ar exit")
        # Cancel the subscribtion when exiting the service to save the expenditure.(退出服务时注销订阅，节省开销)
        try:
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if self.camera_info_sub is not None:
                self.destroy_subscription(self.camera_info_sub)
                self.camera_info_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        response.success = True
        response.message = "exit"
        return response
        
    def set_model_srv_callback(self, request, response):
        # Set model(设置模型)
        with self.lock:
            self.get_logger().info('\033[1;32m%s\033[0m' % "set model {}".format(request.data))
            if request.data == "":
                self.target_model = None
            else:
                self.target_model = request.data
                if self.target_model != 'rectangle':  # If the cube is not being drawn.(如果不是绘制立方体)
                    # Load the model(加载模型)
                    obj = obj_load(os.path.join(MODEL_PATH, self.target_model + '.obj'), swapyz=True)
                    obj.faces = obj.faces[::-1]
                    new_faces = []
                    # Analyze the model and get the point coordinates.(对模型进行解析，获取点坐标)
                    for face in obj.faces:
                        face_vertices = face[0]
                        points = []
                        colors = []
                        for vertex in face_vertices:
                            data = obj.vertices[vertex - 1]
                            points.append(data[:3])
                            if self.target_model != 'cow' and self.target_model != 'wolf':
                                colors.append(data[3:])
                        scale_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * MODELS_SCALE[self.target_model]  # Scale(缩放)
                        points = np.dot(np.array(points), scale_matrix)
                        if self.target_model == 'bicycle':
                            points = np.array([[p[0] - 670, p[1] - 350, p[2]] for p in points])
                            points = R.from_euler('xyz', (0, 0, 180), degrees=True).apply(points)
                        elif self.target_model == 'fox':
                            points = np.array([[p[0], p[1], p[2]] for p in points])
                            points = R.from_euler('xyz', (0, 0, -90), degrees=True).apply(points)
                        elif self.target_model == 'chair':
                            points = np.array([[p[0], p[1], p[2]] for p in points])
                            points = R.from_euler('xyz', (0, 0, -90), degrees=True).apply(points)
                        else:
                            points = np.array([[p[0], p[1], p[2]] for p in points])
                        if len(colors) > 0:
                            color = tuple(255 * np.array(colors[0]))
                        else:
                            color = None
                        new_faces.append((points, color))
                    self.obj = new_faces
        response.success = True
        response.message = "set_model"
        return response

    def camera_info_callback(self, msg):
        # Camera internal parameter callback(摄像头内参信息获取回调)
        with self.lock:
            self.camera_intrinsic = np.array(msg.k).reshape(3, -1)
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, ros_image):
        # Image callback(图像回调)
        # Convert ROS image into numpy format.(将ROS图像消息转化为numpy格式)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        result_image = np.copy(rgb_image)
        with self.lock:
            try:
                # Process image(图像处理)
                result_image = self.image_proc(rgb_image, result_image)
            except Exception as e:
                self.get_logger().info(str(e))
        if self.debug:
            cv2.imshow("result", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        # Convert opencv format into ros format.(opencv格式转为ros格式)
        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))

    def image_proc(self, rgb_image, result_image):
        # Process image(图像处理)
        if self.target_model is not None: 
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # Convert into gray image.(转为灰度图)
            detections = self.tag_detector.detect(gray)  # aprilatg recognition(aprilatg识别)
            if detections != ():
                for detection in detections:  # traverse(遍历)
                    # Acquire four angular points and center point.(获取四个角点和中心)
                    tag_center = detection['center']
                    tag_corners = detection['lb-rb-rt-lt']
                    lb = tag_corners[0]
                    rb = tag_corners[1]
                    rt = tag_corners[2]
                    lt = tag_corners[3]
                    # 绘制四个角点(draw four angular points)
                    cv2.circle(result_image, (int(lb[0]), int(lb[1])), 2, (0, 255, 255), -1)
                    cv2.circle(result_image, (int(lt[0]), int(lt[1])), 2, (0, 255, 255), -1)
                    cv2.circle(result_image, (int(rb[0]), int(rb[1])), 2, (0, 255, 255), -1)
                    cv2.circle(result_image, (int(rt[0]), int(rt[1])), 2, (0, 255, 255), -1)
                    # cv2.circle(result_image, (int(tag_center[0]), int(tag_center[1])), 3, (255, 0, 0), -1)
                    corners = np.array([lb, rb, lt, rt, tag_center]).reshape(5, -1)
                    # Use the world coordinate system k point coordinates (OBJP), the k point coordinates (corners) corresponding to the 2D image coordinate system, and the camera internal parameters camera_intrinsic and dist_coeffs to reverse the external parameters r, t of the picture.
                    # (使用世界坐标系k个点坐标(OBJP)，对应图像坐标系2D的k个点坐标(corners)，以及相机内参camera_intrinsic和dist_coeffs进行反推图片的外参r, t)
                    ret, rvecs, tvecs = cv2.solvePnP(OBJP, corners, self.camera_intrinsic, self.dist_coeffs)
                    if self.target_model == 'rectangle':  # If the cube needs to be displayed, process independently(如果要显示立方体则单独处理)
                        # Backprojection converts world coordinate system points to image points(反向投影将世界坐标系点转换到图像点)
                        imgpts, jac = cv2.projectPoints(AXIS, rvecs, tvecs, self.camera_intrinsic, self.dist_coeffs)
                        result_image = draw_rectangle(result_image, imgpts)
                    else:
                        for points, color in self.obj:
                             dst, jac = cv2.projectPoints(points.reshape(-1, 1, 3)/100.0, rvecs, tvecs, self.camera_intrinsic, self.dist_coeffs)
                             imgpts = dst.astype(int)
                             # Manual coloring(手动上色)
                             if self.target_model == 'cow':
                                 cv2.fillConvexPoly(result_image, imgpts, (0, 255, 255))
                             elif self.target_model == 'wolf':
                                 cv2.fillConvexPoly(result_image, imgpts, (255, 255, 0))
                             else:
                                 cv2.fillConvexPoly(result_image, imgpts, color)

        return result_image

def main():
    node = ARNode('ar_detect')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
