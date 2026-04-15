#!/usr/bin/python3
# coding=utf8
# @Author: Aiden
# @Date: 2024/12/31
import cv2
import math
import numpy as np
from typing import Tuple, Optional, List

def create_rectangle_mask(rect: tuple, shape: tuple) -> np.ndarray:
    """
    将旋转矩形转换为掩码图像

    Args:
        rect: 旋转矩形参数(中心点,尺寸,角度)
        shape: 输出掩码图像的尺寸

    Returns:
        mask: 掩码图像
    """
    box = cv2.boxPoints(rect).astype(int)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [box], -1, 255, thickness=cv2.FILLED)
    return mask


def get_gripper_masks(depth_image: np.ndarray,
                      center: tuple,
                      angle: float,
                      gripper_width: float,
                      gripper_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成夹持器两个方向的掩码

    Args:
        depth_image: 深度图像
        center: 夹持位置中心点
        angle: 夹持角度
        gripper_width: 夹持器宽度
        gripper_height: 夹持器高度

    Returns:
        mask_horizontal: 水平方向掩码
        mask_vertical: 垂直方向掩码
    """
    angle_horizontal = angle
    angle_vertical = angle - 90
    rect_horizontal = (center, (gripper_width, gripper_height), angle_horizontal)
    rect_vertical = (center, (gripper_width, gripper_height), angle_vertical)

    mask_horizontal = create_rectangle_mask(rect_horizontal, depth_image.shape)
    mask_vertical = create_rectangle_mask(rect_vertical, depth_image.shape)
    return mask_horizontal, mask_vertical


def get_obstacle_mask(depth_image: np.ndarray,
                      contour: np.ndarray,
                      plane_values: np.ndarray,
                      max_height: float,
                      obj_height: float,
                      gripper_depth: float) -> np.ndarray:
    """
    生成障碍物掩码

    Args:
        depth_image: 深度图像
        contour: 物体轮廓
        plane_values: 平面方程值
        max_height: 最大障碍物高度
        obj_height: 目标物体高度
        gripper_depth: 夹持器深度

    Returns:
        mask: 障碍物掩码
    """
    # 生成物体掩码
    mask_obj = np.zeros_like(depth_image, dtype=np.uint8)
    cv2.drawContours(mask_obj, [contour], -1, 255, thickness=cv2.FILLED)

    # 生成障碍物掩码
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask[(plane_values > (obj_height - gripper_depth)) &
         (plane_values < max_height) &
         (mask_obj != 255)] = 255

    return mask


def calculate_obj_angles(
        rect_width: float,
        rect_height: float,
        gripper_width: float,
        angle1_info: list,
        angle2_info: list,
) -> list:
    """计算可行的物体抓取角度

    Args:
        rect_angle: 矩形的初始角度
        rect_width: 矩形的宽度
        rect_height: 矩形的高度
        gripper_width: 机械爪的抓取宽度
        angle1_info: 垂直短边的夹取姿态
        angle2_info: 

    Returns:
        可行的抓取角度列表
    """
    # 计算水平和垂直两个抓取角度
    angle1 = angle1_info[0]
    angle2 = angle2_info[0]
    angle1_is_blocked = angle1_info[1]
    angle2_is_blocked = angle2_info[1]

    # 为机械爪预留操作空间
    safe_gripper_width = gripper_width
    possible_angles = []

    # 判断可行的抓取角度
    if max(rect_width, rect_height) < safe_gripper_width:
        if not angle1_is_blocked:
            possible_angles.append(angle1)
        if not angle2_is_blocked:
            possible_angles.append(angle2)
    elif max(rect_width, rect_height) > safe_gripper_width and min(rect_width, rect_height) < safe_gripper_width and not angle2_is_blocked:
        possible_angles = [angle2]
    return possible_angles

def calculate_grasp_angle(position: np.ndarray, angle_list: Optional[List[float]], initial_angle: float) -> Tuple[
    Optional[int], float, float]:
    """
    计算夹持器的偏航角度
    Args:
        position: 目标位置坐标 [x, y]
        angle_list: 可选的角度列表
        initial_angle: 初始角度
    Returns:
        yaw: 计算后的偏航角度
        gripper_angle: 夹持器角度
        initial_angle: 初始角度
    """
    # 计算基础偏航角度
    yaw = math.degrees(math.atan2(position[1], position[0]))

    # 根据象限调整偏航角度
    if position[0] < 0:
        if position[1] < 0:
            yaw += 180
        else:
            yaw -= 180

    gripper_angle = initial_angle

    if angle_list:
        # 应用角度列表中的第一个角度
        yaw += angle_list[0]
        gripper_angle = angle_list[0]

        # 如果有第二个角度选项，计算备选角度
        if len(angle_list) == 2:
            alternative_yaw = yaw + (-90 if yaw > 0 else 90)

            # 选择幅度较小的角度
            if abs(yaw) > abs(alternative_yaw):
                gripper_angle = angle_list[1]
                yaw = alternative_yaw

        # 将角度映射到控制范围
        yaw = 500 + int(yaw / 240 * 1000)
    else:
        yaw = None

    return yaw, gripper_angle

