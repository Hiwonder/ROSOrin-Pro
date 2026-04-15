#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2022/10/22
import cv2
import math
import yaml
import numpy as np
import transforms3d as tfs
from rclpy.node import Node
from geometry_msgs.msg import Pose, Quaternion

range_rgb = {  
    'red': (255, 0, 50),     
    'blue': (0, 50, 255),   
    'green': (0, 255, 50),   
    'black': (0, 0, 0),      
    'white': (255, 255, 255)  
}

def loginfo(msg):
    Node.get_logger().info('\033[1;32m%s\033[0m' % msg)

def val_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def empty_func(img=None):
    return img

def set_range(x, x_min, x_max):
    tmp = x if x > x_min else x_min
    tmp = tmp if tmp < x_max else x_max
    return tmp

def get_yaml_data(yaml_file):
    yaml_file = open(yaml_file, 'r', encoding='utf-8')
    file_data = yaml_file.read()
    yaml_file.close()

    data = yaml.load(file_data, Loader=yaml.FullLoader)

    return data

def save_yaml_data(data, yaml_file):
    f = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(data, f)

    f.close()

def distance(point_1, point_2):
    """
    calculate the distance between two points. 计算两个点间的距离
    :param point_1: Point 1. 点1
    :param point_2: Point 2. 点2
    :return: Distance between two points. 两点间的距离
    """
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

def box_center(box):
    """
    Calculate the center of quadrangle box. 计算四边形box的中心
    :param box: box （x1, y1, x2, y2)形式(box （x1, y1, x2, y2)type)
    :return:  Center coordinate. 中心坐标（x, y)
    """
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def bgr8_to_jpeg(value, quality=75):
    """
    Convert data in the format of cv bgr8 into jpg. 将cv bgr8格式数据转换为jpg格式
    :param value: Original data 原始数据
    :param quality:  jpg quality. Maximum value is 100. jpg质量 最大值100
    :return:
    """
    return bytes(cv2.imencode('.jpg', value)[1])


def point_remapped(point, now, new, data_type=float):
    """
    Map the coordinate of one point from a picture to a new picture of different size. 将一个点的坐标从一个图片尺寸映射的新的图片上
    :param point: Coordinate of point.  点的坐标
    :param now: Size of current picture. 现在图片的尺寸
    :param new: New picture size. 新的图片尺寸
    :return: New point coordinate. 新的点坐标
    """
    x, y = point
    now_w, now_h = now
    new_w, new_h = new
    new_x = x * new_w / now_w
    new_y = y * new_h / now_h
    return data_type(new_x), data_type(new_y)


def get_area_max_contour(contours, threshold=50):
    """
    Get the contour whose area is the largest. Filter out those whose area is too small. 获取轮廓中面积最重大的一个, 过滤掉面积过小的情况
    :param contours: Contour list 轮廓列表
    :param threshold: Area threshold. Contour whose area is less than this value will be filtered out. 面积阈值, 小于这个面积的轮廓会被过滤
    :return: If the maximum contour area is greater than this threshold, return the largest contour, otherwise return None如果最大的轮廓面积大于阈值则返回最大的轮廓, 否则返回None
    """
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # Iterate through all contours 历遍所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # Compute the area of each contour 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > threshold:  # Filter out noise or irrelevant contours 过滤干扰
                area_max_contour = c

    return area_max_contour,  contour_area_max  # Return the largest contour 返回最大的轮廓


def vector_2d_angle(v1, v2):
    """
    Calculate the angle between two vectors -pi ~ pi. 计算两向量间的夹角 -pi ~ pi
    :param v1: first vector 第一个向量
    :param v2: second vector 第二个向量
    :return: angle 角度
    """
    d_v1_v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = v1.dot(v2) / (d_v1_v2)
    sin = np.cross(v1, v2) / (d_v1_v2)
    angle = np.degrees(np.arctan2(sin, cos))
    return angle


def warp_affine(image, points, scale=1.0):
    """
     Simple alignment. Calculate the angle of the line connecting the two points. Rotate the picture around the image
     center to make the line horizontal can be used to align the face简单的对齐，计算两个点的连线的角度，以图片中心为原点旋转图片，使线水平
    可以用在人脸对齐上

    :param image: select face picture 要选择的人脸图片
    :param points: coordinate of two points 两个点的坐标 ((x1, y1), (x2, y2))
    :param scale: scaling 缩放比例
    :return: rotated picture 旋转后的图片
    """
    w, h = image.shape[:2]
    dy = points[1][1] - points[0][1]
    dx = points[1][0] - points[0][0]
    # Calculate the rotation angle and rotate picture. 计算旋转角度并旋转图片
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, scale=scale)
    return cv2.warpAffine(image, rot, dsize=(h, w))

def perspective_transform(img, src, dst, debug=False):
    """
    perform perspective transformation: converting a skewed view of a road image taken from an oblique angle into a top-down
    view, aligning the camera's viewpoint parallel to the road. 执行透视变换：将倾斜视角拍摄到的道路图像转换成鸟瞰图，即将摄像机的视角转换到和道路平行。
    :param img: input image 输入图像
    :param src: the coordinates of the four points of the rectangle to be measured in the source image 源图像中待测矩形的四点坐标
    :param dst: the coordinates of the four points of the rectangle in the target image 目标图像中矩形的四点坐标
    """
    img_size = (img.shape[1], img.shape[0])
    # Manually extract the vertices for performing the perspective transformation. 手动提取用于执行透视变换的顶点
    '''
    # left_down
    # left_up
    # right_up
    # right_down
    src = np.float32(
        [[89, 370],
         [128, 99],
         [436, 99],
         [472, 371]])
    
    dst = np.float32(
        [[65, 430],
         [65, 55],
         [575,55],
         [575,430]])
    '''
    m = cv2.getPerspectiveTransform(src, dst)  # Calculate the perspective transformation matrix. 计算透视变换矩阵
    if debug:
        m_inv = cv2.getPerspectiveTransform(dst, src)
    else:
        m_inv = None
    # Perform perspective transformation with parameters: input image, output image, target image size, and the cv2.INTER_LINEAR
    # interpolation method. 进行透视变换 参数：输入图像、输出图像、目标图像大小、cv2.INTER_LINEAR插值方法
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    #unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # debugging 调试

    return warped, m, m_inv

def pixels_to_world(pixels, K, T):
    """
    Convert the pixel coordinates to world coordinates. 像素坐标转世界坐标
    Pixel coordinates list. pixels像素坐标列表
    Intrinsic camera parameters matrix K. K 相机内参 np 矩阵
    Extrinsic camera parameters matrix T. T 相机外参 np 矩阵
    """
    invK = K.I
    t, r, _, _ =  tfs.affines.decompose(np.matrix(T))
    invR = np.matrix(r).I
    R_inv_T = np.dot(invR, np.matrix(t).T)
    world_points = []
    for p in pixels:
        coords = np.float64([p[0], p[1], 1.0]).reshape(3, 1)
        cam_point = np.dot(invK, coords)
        world_point = np.dot(invR, cam_point)
        scale = R_inv_T[2][0] / world_point[2][0]
        scale_world = np.multiply(scale, world_point)
        world_point = np.array((np.asmatrix(scale_world) - np.asmatrix(R_inv_T))).reshape(-1,)
        world_points.append(world_point)
    return world_points

def world_to_pixels(world_points, K, T):
    """
    Convert 3D world coordinates to 2D pixel coordinates. 将世界坐标点转换为像素坐标
    Args:
        world_points: A list of 3D points in world coordinates. 世界坐标点列表
        K: Camera intrinsic matrix. 相机内参矩阵
        T: Camera extrinsic matrix. 外参矩阵 [R|t]
    Returns:
        pixel_points: A list of 2D pixel coordinates. 像素坐标点列表
    """
    pixel_points = []
    for wp in world_points:
        # Convert to homogeneous coordinates. 将世界坐标转换为齐次坐标
        world_homo = np.append(wp, 1).reshape(4, 1)
        # Transform to camera coordinates using extrinsic matrix. 通过外参矩阵转换到相机坐标系
        camera_point = np.dot(T, world_homo)
        # Project to image plane. 投影到像素平面
        pixel_homo = np.dot(K, camera_point[:3])
        # Normalization 归一化
        pixel = (pixel_homo / pixel_homo[2])[:2].reshape(-1)
        pixel_points.append(pixel)
    return pixel_points

def calculate_pixel_length(world_length, K, T):
    """
    Compute the corresponding pixel length for a given length in world coordinates. 计算世界坐标中的长度在像素坐标中的对应长度
    Args:
        world_length: Length in world space 世界坐标中的长度
        K: Camera intrinsic matrix 相机内参矩阵
        T: Extrinsic transformation matrix 外参矩阵
    Returns:
        pixel_length: Corresponding length in pixel space 像素坐标中的长度
    """
    # Define a starting point and direction. 定义起始点和方向
    start_point = np.array([0, 0, 0])  # Starting point 起始点
    direction = np.array([0, 1, 0])  # y-direction y方向

    # Compute the endpoint in world coordinates. 计算终点坐标
    end_point = start_point + direction * world_length
    # Transform both endpoints to pixel coordinates. 转换两个端点到像素坐标
    pixels = world_to_pixels([start_point, end_point], K, T)
    # Calculate Euclidean distance in pixel space. 计算像素距离
    pixel_length = np.linalg.norm(pixels[1] - pixels[0])
    
    return int(pixel_length)

def extristric_plane_shift(tvec, rmat, delta_z):
    delta_t = np.array([[0], [0], [delta_z]])
    tvec_new = tvec + np.dot(rmat, delta_t)
    return tvec_new, rmat

pixel_to_world = pixels_to_world

def ros_pose_to_list(pose):
    t = np.asarray([pose.position.x, pose.position.y, pose.position.z])
    q = np.asarray([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    return t, q

def draw_tags(image, tags, corners_color=(0, 125, 255), center_color=(0, 255, 0)):
    for tag in tags:
        corners = tag.corners.astype(int)
        center = tag.center.astype(int)
        # cv2.putText(image, "%d"%tag.tag_id, (int(center[0] - (7 * len("%d"%tag.tag_id))), int(center[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if corners_color is not None:
            for p in corners:
                cv2.circle(image, tuple(p.tolist()), 5, corners_color, -1)
        if center_color is not None:
            cv2.circle(image, tuple(center.tolist()), 8, center_color, -1)
    return image

def show_faces(detect_img, result_img, boxes, landmarks, bbox_color=(0, 255, 0), ll_color=(0, 0, 255)):
    """Draw bounding boxes and face landmarks on image."""
    detect_size = detect_img.shape[:2]
    show_size = result_img.shape[:2]
    for bb, ll in zip(boxes, landmarks):
        p1 = point_remapped(bb[:2], detect_size, show_size, data_type=int)
        p2 = point_remapped(bb[2:4], detect_size, show_size, data_type=int)
        cv2.rectangle(result_img, p1, p2, bbox_color, 2)
        for i, p in enumerate(ll):
            x, y = point_remapped(p, detect_size, show_size, data_type=int)
            cv2.circle(result_img, (x, y), 2, ll_color, 2)
    return result_img


def mp_face_location(results, img):
    h, w, c, = img.shape
    boxes = []
    keypoints = []
    if results.detections:
        for detection in results.detections:
            x_min = detection.location_data.relative_bounding_box.xmin
            y_min = detection.location_data.relative_bounding_box.ymin
            width = detection.location_data.relative_bounding_box.width
            height = detection.location_data.relative_bounding_box.height
            x_min, y_min = max(x_min * w, 0), max(y_min * h, 0)
            x_max, y_max = min(x_min + width * w, w), min(y_min + height * h, h)
            boxes.append((x_min, y_min, x_max, y_max))
            relative_keypoints = detection.location_data.relative_keypoints
            keypoints.append([(point.x * w, point.y * h) for point in relative_keypoints])
    return boxes, keypoints
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def qua2rpy(qua):
    if type(qua) == Quaternion:
        x, y, z, w = qua.x, qua.y, qua.z, qua.w
    else:
        x, y, z, w = qua[0], qua[1], qua[2], qua[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - x * z), -1.0, 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
  
    return roll, pitch, yaw

def rpy2qua(roll, pitch, yaw):
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    cp = math.cos(pitch*0.5)
    sp = math.sin(pitch*0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    q = Pose()
    q.orientation.w = cy * cp * cr + sy * sp * sr
    q.orientation.x = cy * cp * sr - sy * sp * cr
    q.orientation.y = sy * cp * sr + cy * sp * cr
    q.orientation.z = sy * cp * cr - cy * sp * sr
    return q.orientation

def xyz_quat_to_mat(xyz, quat):
    mat = tfs.quaternions.quat2mat(np.asarray(quat))
    mat = tfs.affines.compose(np.squeeze(np.asarray(xyz)), mat, [1, 1, 1])
    return mat

def xyz_rot_to_mat(xyz, rot):
    return np.row_stack((np.column_stack((rot, xyz)), np.array([[0, 0, 0, 1]])))

def xyz_euler_to_mat(xyz, euler, degrees=True):
    if degrees:
        mat = tfs.euler.euler2mat(math.radians(euler[0]), math.radians(euler[1]), math.radians(euler[2]))
    else:
        mat = tfs.euler.euler2mat(euler[0], euler[1], euler[2])
    mat = tfs.affines.compose(np.squeeze(np.asarray(xyz)), mat, [1, 1, 1])
    return mat

def mat_to_xyz_euler(mat, degrees=True):
    t, r, _, _ = tfs.affines.decompose(mat)
    if degrees:
        euler = np.degrees(tfs.euler.mat2euler(r))
    else:
        euler = tfs.euler.mat2euler(r)
    return t, euler


