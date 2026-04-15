#!/usr/bin/python3
# coding=utf8
# @Author: Aiden
# @Date: 2024/12/31
import cv2
import copy
import math
import numpy as np
from sdk import common
from typing import Tuple, List, Optional

# 机械臂夹持器的物理参数(单位:米)
GRIPPER_HB = 0.014  
GRIPPER_BC = 0.03 
GRIPPER_ED = 0.037
GRIPPER_DC = 0.022
EDC = math.radians(180 - 21)
GRIPPER_IH = 0.02
GRIPPER_IG = 0.005
LCD = math.acos((GRIPPER_HB - GRIPPER_IG) / GRIPPER_IH)
GRIPPER_EC = (GRIPPER_ED**2 + GRIPPER_DC**2 - 2 * GRIPPER_ED * GRIPPER_DC * math.cos(EDC)) ** 0.5
ECD = math.acos((GRIPPER_DC**2 + GRIPPER_EC**2 - GRIPPER_ED**2) / (2 * GRIPPER_DC * GRIPPER_EC))

def get_gripper_size(angle: float, angle_zero=200) -> Tuple[float, float]:
    """
    根据夹持器角度计算夹持器的宽度和高度
    
    Args:
        angle: 夹持器角度(0-1000)
        angle_zero: 夹持器角度零点(默认200)
    Returns:
        width: 夹持器宽度
        height: 夹持器高度
    """
    angle = math.radians((angle - angle_zero) / 1000 * 180)
     
    GRIPPER_BJ = math.cos(angle) * GRIPPER_BC
    GRIPPER_HJ = GRIPPER_HB + GRIPPER_BJ
    GRIPPER_LC = math.cos(LCD + ECD) * GRIPPER_EC
    GRIPPER_KE = GRIPPER_HJ - GRIPPER_LC

    GRIPPER_KE = GRIPPER_HB + math.cos(angle) * GRIPPER_BC - math.cos(LCD + ECD) * GRIPPER_EC

    GRIPPER_JC = (GRIPPER_BC**2 - GRIPPER_BJ**2)**0.5
    GRIPPER_LE = (GRIPPER_EC**2 - GRIPPER_LC**2)**0.5
    gripper_depth = GRIPPER_JC + GRIPPER_LE
    gripper_width = 2*GRIPPER_KE
    
    return gripper_width, gripper_depth

def set_gripper_size(width: float) -> int:
    """
    根据目标宽度计算夹持器需要的角度
    
    Args:
        width: 目标宽度(米)
        
    Returns:
        angle: 夹持器角度(0-1000)
    """
    width = width / 2
    a = (width - GRIPPER_HB + math.cos(LCD + ECD) * GRIPPER_EC) / GRIPPER_BC
    a = max(-1.0, min(1.0, a))
    return int(math.degrees(math.acos(a)) / 180 * 1000 + 200)

def world_to_pixels(world_points, K, T):
    """
    将世界坐标点转换为像素坐标
    Args:
        world_points: 世界坐标点列表
        K: 相机内参矩阵
        T: 外参矩阵 [R|t]
    Returns:
        pixel_points: 像素坐标点列表
    """
    pixel_points = []
    for wp in world_points:
        # 将世界坐标转换为齐次坐标
        world_homo = np.append(wp, 1).reshape(4, 1)
        # 通过外参矩阵转换到相机坐标系
        camera_point = np.dot(T, world_homo)
        # 投影到像素平面
        pixel_homo = np.dot(K, camera_point[:3])
        # 归一化
        pixel = (pixel_homo / pixel_homo[2])[:2].reshape(-1)
        pixel_points.append(pixel)
    return pixel_points

def calculate_pixel_length(world_length, K, T):
    """
    计算世界坐标中的长度在像素坐标中的对应长度
    Args:
        world_length: 世界坐标中的长度
        K: 相机内参矩阵
        T: 外参矩阵
    Returns:
        pixel_length: 像素坐标中的长度
    """
    # 定义起始点和方向
    start_point = np.array([0, 0, 0])  # 起始点
    direction = np.array([0, 1, 0])  # y方向

    # 计算终点坐标
    end_point = start_point + direction * world_length
    # 转换两个端点到像素坐标
    pixels = world_to_pixels([start_point, end_point], K, T)
    # 计算像素距离
    pixel_length = np.linalg.norm(pixels[1] - pixels[0])

    return int(pixel_length)

def get_plane_values(depth_image: np.ndarray, 
                    plane: Tuple[float, float, float, float],
                    intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    计算深度图像中每个点到平面的距离
    
    Args:
        depth_image: 深度图像
        plane: 平面方程参数(a,b,c,d)
        intrinsic_matrix: 相机内参矩阵
        
    Returns:
        plane_values: 每个点到平面的距离
    """
    a, b, c, d = plane
    # 提取相机内参
    fx = intrinsic_matrix[0]
    fy = intrinsic_matrix[4]
    cx = intrinsic_matrix[2]
    cy = intrinsic_matrix[5]
    
    # 图像尺寸
    H, W = depth_image.shape
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 计算相机坐标
    z = depth_image / 1000.0  # 转换为米
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 计算平面方程值
    plane_values = a * x + b * y + c * z + d
    
    return plane_values

def create_roi_mask(
    depth_image: np.ndarray,
    bgr_image: np.ndarray,
    corners: np.ndarray,
    camera_info: object,
    extrinsic: np.ndarray,
    max_height: float,
    max_obj_height: float,
) -> np.ndarray:
    """
    创建感兴趣区域(ROI)的遮罩
    Args:
        depth_image: 深度图像
        bgr_image: BGR图像
        corners: 角点坐标
        camera_info: 相机参数
        extrinsic: 外参矩阵
        max_height: 最大高度
        max_obj_height: 物体最大高度
    Returns:
        mask: ROI遮罩
    """
    # image_height, image_width = depth_image.shape[:2]
    
    # # 分解外参矩阵
    # translation_vec = extrinsic[:1]
    # rotation_mat = extrinsic[1:]
    # corners_copy = copy.deepcopy(corners)
    
    # # 投影中心点
    # center_points, _ = cv2.projectPoints(
    #     corners_copy[-1:],
    #     np.array(rotation_mat),
    #     np.array(translation_vec),
    #     np.matrix(camera_info.k).reshape(1, -1, 3),
    #     np.array(camera_info.d)
    # )
    # center_points = np.int32(center_points).reshape(2)

    # # 计算平面偏移后的外参
    # shifted_tvec, shifted_rmat = common.extristric_plane_shift(
    #     np.array(translation_vec).reshape((3, 1)),
    #     np.array(rotation_mat),
    #     max_obj_height
    # )
    
    # # 投影其他角点
    # projected_points, _ = cv2.projectPoints(
    #     corners_copy[:-1],
    #     np.array(shifted_rmat),
    #     np.array(shifted_tvec),
    #     np.matrix(camera_info.k).reshape(1, -1, 3),
    #     np.array(camera_info.d)
    # )
    # projected_points = np.int32(projected_points).reshape(-1, 2)
    
    # # 计算ROI边界
    # x_min = max(0, min(projected_points[:, 0]))
    # x_max = min(image_width, max(projected_points[:, 0]))
    # y_min = max(0, min(projected_points[:, 1]))
    # y_max = min(image_height, max(projected_points[:, 1]))
   
    # # 在BGR图像上绘制ROI框
    # # cv2.rectangle(bgr_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # # 创建ROI区域
    # x, y = x_min + 10, y_min - 40
    # w, h = x_max - x_min, y_max - y_min
    
    # # 创建遮罩
    # mask = np.zeros_like(depth_image)
    # x2 = min(x + w, image_width)
    # y2 = max(y, 0)
    # mask[y2:y+h, x:x2] = depth_image[y2:y+h, x:x2]

    # # 将深度图像中对应的区域外设置为0
    # depth_image[mask == 0] = max_height

    # return depth_image

    # 深拷贝避免上层意外修改原始数据
    full_depth = depth_image.copy()

    # 把无效点统一填成 max_height，防止后续算法将其当作最近点
    full_depth[full_depth == 0] = max_height

    return full_depth

def find_depth_range(depth_image: np.ndarray, max_distance: float) -> Tuple[float, float]:
    """
    查找深度图像中的最小
    Args:
        depth_image: 深度图像
    Returns:
        min_distance: 最小距离(mm)
    """
    height, width = depth_image.shape[:2]
    
    # 处理深度数据
    depth = np.copy(depth_image).reshape(-1)
    depth[depth <= 0] = max_distance  # 将无效值设为max_distance
    
    # 找到最近点
    min_idx = np.argmin(depth)
    min_y, min_x = min_idx // width, min_idx % width
    min_distance = depth_image[min_y, min_x] 
    
    return min_distance

def extract_contours(
    plane_values: np.ndarray,
    filter_height: float
) -> List[np.ndarray]:
    """
    提取深度图像中的轮廓
    Args:
        plane_values: 平面值
        filter_height: 过滤高度
    Returns:
        contours: 轮廓列表
    """
    # 过滤高度
    filtered_image = np.where(plane_values <= filter_height, 0, 255).astype(np.uint8)
    
    # 二值化和轮廓提取
    _, binary = cv2.threshold(filtered_image, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow(color, binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def convert_depth_to_camera_coords(
    pixel_coords: Tuple[float, float, float],
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    将深度像素坐标转换为相机坐标系
    Args:
        pixel_coords: 像素坐标 (x, y, z)
        intrinsic_matrix: 相机内参矩阵
    Returns:
        camera_coords: 相机坐标系下的坐标
    """
    fx, fy = intrinsic_matrix[0], intrinsic_matrix[4]
    cx, cy = intrinsic_matrix[2], intrinsic_matrix[5]
    px, py, pz = pixel_coords
    
    x = (px - cx) * pz / fx
    y = (py - cy) * pz / fy
    z = pz
    
    return np.array([x, y, z])

def calculate_world_position(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    plane: Tuple[float, float, float, float],
    endpoint: np.ndarray,
    hand2cam_tf_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray,
    camera_roll_offset: float = 6.0,  #0.96  0.0   前后误差修改
    camera_pitch_offset: float = 0.0, #-0.73
    camera_yaw_offset: float = 0.0
) -> np.ndarray:
    """
    计算世界坐标系中的位置 (消除摄像头 Roll, Pitch, Yaw 倾斜)

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标
        depth: 深度值 (mm)
        plane: 平面参数 (a, b, c, d)
        endpoint: 末端执行器在世界坐标系中的位姿矩阵 (4x4)
        hand2cam_tf_matrix: 手眼标定矩阵，将相机坐标系转换到末端执行器坐标系 (4x4)
        intrinsic_matrix: 相机内参矩阵 (3x3)
        camera_roll_offset: 相机 Roll 轴的倾斜角度 (度)
        camera_pitch_offset: 相机 Pitch 轴的倾斜角度 (度)
        camera_yaw_offset: 相机 Yaw 轴的倾斜角度 (度)

    Returns:
        world_position: 世界坐标系中的位置
    """

    # 1. 消除相机倾斜：构建旋转矩阵
    roll_offset_rad = math.radians(camera_roll_offset)
    pitch_offset_rad = math.radians(camera_pitch_offset)
    yaw_offset_rad = math.radians(camera_yaw_offset)

    # 构建绕 X 轴旋转的矩阵 (Roll)
    rotation_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll_offset_rad), -math.sin(roll_offset_rad)],
        [0, math.sin(roll_offset_rad), math.cos(roll_offset_rad)]
    ])

    # 构建绕 Y 轴旋转的矩阵 (Pitch)
    rotation_y = np.array([
        [math.cos(pitch_offset_rad), 0, math.sin(pitch_offset_rad)],
        [0, 1, 0],
        [-math.sin(pitch_offset_rad), 0, math.cos(pitch_offset_rad)]
    ])

    # 构建绕 Z 轴旋转的矩阵 (Yaw)
    rotation_z = np.array([
        [math.cos(yaw_offset_rad), -math.sin(yaw_offset_rad), 0],
        [math.sin(yaw_offset_rad), math.cos(yaw_offset_rad), 0],
        [0, 0, 1]
    ])

    #  组合旋转矩阵 (注意旋转顺序)
    # ZYX (Yaw, Pitch, Roll)
    camera_rotation_correction_matrix = np.eye(4)  # 初始化为单位矩阵
    camera_rotation_correction_matrix[:3, :3] = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    # 2. 将像素坐标转换为相机坐标系
    camera_z = depth / 1000.0  # 转换为米
    camera_x = (pixel_x - intrinsic_matrix[2]) * camera_z / intrinsic_matrix[0]
    camera_y = (pixel_y - intrinsic_matrix[5]) * camera_z / intrinsic_matrix[4]
    camera_position = np.array([camera_x, camera_y, camera_z, 1])  # 齐次坐标

    # 3. 消除相机倾斜：将相机坐标转换到未倾斜的相机坐标系
    # 注意：hand2cam_tf_matrix 已经将坐标转换到了 *倾斜的* 相机坐标系
    #   我们需要先将坐标转换回 *未倾斜的* 相机坐标系
    camera_position_corrected = np.dot(camera_rotation_correction_matrix, camera_position)

    # 4. 将（校正后的）相机坐标系转换到末端执行器坐标系
    hand_position = np.dot(hand2cam_tf_matrix, camera_position_corrected)

    # 5. 将末端执行器坐标系转换到世界坐标系
    world_position_homo = np.dot(endpoint, hand_position)
    world_position = world_position_homo[:3] / world_position_homo[3] 

    return world_position

