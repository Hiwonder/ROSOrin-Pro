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

# Physical parameters of the robotic gripper with units in meters. 机械臂夹持器的物理参数(单位:米)
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
    Compute the gripper's width and height based on its opening angle. 根据夹持器角度计算夹持器的宽度和高度
    
    Args:
        angle: Gripper opening angle with a range from 0 to 1000. 夹持器角度(0-1000)
        angle_zero: Zero reference angle for the gripper, set to 200 by default). 夹持器角度零点(默认200)
    Returns:
        width: Gripper width 夹持器宽度
        height: Gripper height 夹持器高度
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
    Compute the gripper angle required for a target width. 根据目标宽度计算夹持器需要的角度
    
    Args:
        width: Target width in meters. 目标宽度(米)
        
    Returns:
        angle: Gripper angle (range: 0–1000). 夹持器角度(0-1000)
    """
    width = width / 2
    a = (width - GRIPPER_HB + math.cos(LCD + ECD) * GRIPPER_EC) / GRIPPER_BC
    a = max(-1.0, min(1.0, a))
    return int(math.degrees(math.acos(a)) / 180 * 1000 + 200)

def world_to_pixels(world_points, K, T):
    """
    Convert world coordinates to pixel coordinates. 将世界坐标点转换为像素坐标
    Args:
        world_points: List of world coordinates. 世界坐标点列表
        K: Camera intrinsic matrix 相机内参矩阵
        T: Extrinsic transformation matrix [R|t] 外参矩阵 [R|t]
    Returns:
        pixel_points: List of corresponding pixel coordinates 像素坐标点列表
    """
    pixel_points = []
    for wp in world_points:
        # Convert world coordinates to homogeneous coordinates. 将世界坐标转换为齐次坐标
        world_homo = np.append(wp, 1).reshape(4, 1)
        # Transform to camera coordinates using the extrinsic matrix. 通过外参矩阵转换到相机坐标系
        camera_point = np.dot(T, world_homo)
        # Project onto the image plane using the intrinsic matrix. 投影到像素平面
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
    start_point = np.array([0, 0, 0])  # starting point 起始点
    direction = np.array([0, 1, 0])  # y-direction y方向

    # Compute the endpoint. 计算终点坐标
    end_point = start_point + direction * world_length
    # Transform both endpoints to pixel coordinates. 转换两个端点到像素坐标
    pixels = world_to_pixels([start_point, end_point], K, T)
    # Calculate Euclidean distance in pixel space. 计算像素距离
    pixel_length = np.linalg.norm(pixels[1] - pixels[0])

    return int(pixel_length)

def get_plane_values(depth_image: np.ndarray, 
                    plane: Tuple[float, float, float, float],
                    intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the distance from each depth pixel to a given plane. 计算深度图像中每个点到平面的距离
    
    Args:
        depth_image: Depth image 深度图像
        plane: Plane equation parameters 平面方程参数(a,b,c,d)
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
        
    Returns:
        plane_values: Distance from each pixel to the plane 每个点到平面的距离
    """
    a, b, c, d = plane
    # Extract camera intrinsics 提取相机内参
    fx = intrinsic_matrix[0]
    fy = intrinsic_matrix[4]
    cx = intrinsic_matrix[2]
    cy = intrinsic_matrix[5]
    
    # Image dimensions 图像尺寸
    H, W = depth_image.shape
    
    # Generate pixel coordinate grid 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Calculate the camera coordinates 计算相机坐标
    z = depth_image / 1000.0  # Convert units to meters 转换为米
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Evaluate the plane equation values 计算平面方程值
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
    Create a region of interest (ROI) mask. 创建感兴趣区域(ROI)的遮罩
    Args:
        depth_image: Depth image 深度图像
        bgr_image: BGR image BGR图像
        corners: corner coordinates 角点坐标
        camera_info: Camera intrinsic parameters 相机参数
        extrinsic: Camera extrinsic matrix 外参矩阵
        max_height: Maximum threshold height 最大高度
        max_obj_height: Maximum object height 物体最大高度
    Returns:
        mask: ROI mask ROI遮罩
    """
    image_height, image_width = depth_image.shape[:2]
    
    # Decompose the extrinsic matrix 分解外参矩阵
    translation_vec = extrinsic[:1]
    rotation_mat = extrinsic[1:]
    corners_copy = copy.deepcopy(corners)
    
    # Project the central point 投影中心点
    center_points, _ = cv2.projectPoints(
        corners_copy[-1:],
        np.array(rotation_mat),
        np.array(translation_vec),
        np.matrix(camera_info.k).reshape(1, -1, 3),
        np.array(camera_info.d)
    )
    center_points = np.int32(center_points).reshape(2)

    # Compute new extrinsic matrix after applying plane offset 计算平面偏移后的外参
    shifted_tvec, shifted_rmat = common.extristric_plane_shift(
        np.array(translation_vec).reshape((3, 1)),
        np.array(rotation_mat),
        max_obj_height
    )
    
    # Project other ROI corner points 投影其他角点
    projected_points, _ = cv2.projectPoints(
        corners_copy[:-1],
        np.array(shifted_rmat),
        np.array(shifted_tvec),
        np.matrix(camera_info.k).reshape(1, -1, 3),
        np.array(camera_info.d)
    )
    projected_points = np.int32(projected_points).reshape(-1, 2)
    
    # Calculate the bounding box for the ROI 计算ROI边界
    x_min = max(0, min(projected_points[:, 0]))
    x_max = min(image_width, max(projected_points[:, 0]))
    y_min = max(0, min(projected_points[:, 1]))
    y_max = min(image_height, max(projected_points[:, 1]))
   
    # Draw the ROI box on the BGR image 在BGR图像上绘制ROI框
    # cv2.rectangle(bgr_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Create ROI are 创建ROI区域
    x, y = x_min + 10, y_min - 40
    w, h = x_max - x_min, y_max - y_min
    
    # create mask 创建遮罩
    mask = np.zeros_like(depth_image)
    x2 = min(x + w, image_width)
    y2 = max(y, 0)
    mask[y2:y+h, x:x2] = depth_image[y2:y+h, x:x2]

    # Zero out all regions outside the ROI in the depth image 将深度图像中对应的区域外设置为0
    depth_image[mask == 0] = max_height

    return depth_image

def find_depth_range(depth_image: np.ndarray, max_distance: float) -> Tuple[float, float]:
    """
    Find the minimum distance in a depth image. 查找深度图像中的最小
    Args:
        depth_image: Depth image 深度图像
    Returns:
        min_distance: Minimum distance in millimeters. 最小距离(mm)
    """
    height, width = depth_image.shape[:2]
    
    # Process depth data 处理深度数据
    depth = np.copy(depth_image).reshape(-1)
    depth[depth <= 0] = max_distance  # Set invalid values to max_distance 将无效值设为max_distance
    
    # Find the closest point 找到最近点
    min_idx = np.argmin(depth)
    min_y, min_x = min_idx // width, min_idx % width
    min_distance = depth_image[min_y, min_x] 
    
    return min_distance

def extract_contours(
    plane_values: np.ndarray,
    filter_height: float
) -> List[np.ndarray]:
    """
    Extract contours from a depth image. 提取深度图像中的轮廓
    Args:
        plane_values: Plane value 平面值
        filter_height: Height threshold for filtering 过滤高度
    Returns:
        contours: List of extracted contours 轮廓列表
    """
    # Apply height threshold 过滤高度
    filtered_image = np.where(plane_values <= filter_height, 0, 255).astype(np.uint8)
    
    # Perform binarization and contour extraction 二值化和轮廓提取
    _, binary = cv2.threshold(filtered_image, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow(color, binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def convert_depth_to_camera_coords(
    pixel_coords: Tuple[float, float, float],
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert depth pixel coordinates to camera coordinate system. 将深度像素坐标转换为相机坐标系
    Args:
        pixel_coords: Pixel coordinates 像素坐标 (x, y, z)
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
    Returns:
        camera_coords: Coordinates in the camera coordinate system 相机坐标系下的坐标
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
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute the position in the world coordinate system. 计算世界坐标系中的位置
    Args:
        pixel_x: Pixel x-coordinate 像素x坐标
        pixel_y: Pixel y-coordinate 像素y坐标
        depth: Depth value 深度值
        plane: Plane parameters 平面参数 (a, b, c, d)
        endpoint: End-effector pose 末端执行器位姿
        hand2cam_tf_matrix: Hand–eye transformation matrix 手眼变换矩阵
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
    Returns:
        world_position: Position in the world coordinate system 世界坐标系中的位置
    """
    # Transform to camera coordinate system   转换到相机坐标系
    camera_position = convert_depth_to_camera_coords(
        [pixel_x, pixel_y, depth / 1000],
        intrinsic_matrix
    )
  
    #  Compute end-effector pose. 计算末端执行器位姿
    pose_end = np.matmul(
        hand2cam_tf_matrix,
            common.xyz_euler_to_mat(camera_position, (0, 0, 0))
    )
    
    # Transform to world coordinate system. 转换到世界坐标系
    # world_pose = np.matmul(endpoint ,pose_end)
    # world_position, euler = common.mat_to_xyz_euler(world_pose)
    world_position = (endpoint + pose_end)[:3, 3]
    
    world_position = [world_position[-1], world_position[1], world_position[0]]
    
    # Use the plane equation to compute the z-coordinate. 使用平面方程计算z坐标
    a, b, c, d = plane
    world_position[2] = (
        camera_position[0] * a +
        camera_position[1] * b +
        camera_position[2] * c + d
    )
    
    return world_position


