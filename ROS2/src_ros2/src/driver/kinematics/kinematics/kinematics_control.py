#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/21
# @author:aiden
# 机械臂运动学调用(call robotic arm kinematics)
from kinematics_msgs.srv import SetRobotPose, SetJointValue

def set_pose_target(position, pitch, pitch_range=[-180.0, 180.0], resolution=1.0):
    '''
    Given a coordinate and a pitch angle, return the inverse kinematics solution(给定坐标和俯仰角，返回逆运动学解)
    position: The target position in a list [x, y, z] with units of meters(目标位置，列表形式[x, y, z]，单位m)
    pitch: The target pitch angle in degrees, ranging from -180 to 180(目标俯仰角，单位度，范围-180~180)
    pitch_range: If a solution cannot be found at the target pitch angle, search for a solution within this range(如果在目标俯仰角找不到解，则在这个范围内寻找解)
    resolution: The resolution of the pitch_range in degrees(pitch_range范围角度的分辨率)
    return: Whether the call is successful, the target positions of the servo, the current positions of the servo, the target posture of the robotic arm, and the changes in the rotation of all servos for the optimal solution(调用是否成功， 舵机的目标位置， 当前舵机的位置， 机械臂的目标姿态， 最优解所有舵机转动的变化量)
    '''
    msg = SetRobotPose.Request()
    msg.position = [float(i) for i in position]
    msg.pitch = float(pitch)
    msg.pitch_range = [float(i) for i in pitch_range]
    msg.resolution = float(resolution)
    return msg

def set_joint_value_target(joint_value):
    '''
    Given the rotation angles of each servo, return the target position and posture of the robotic arm(给定每个舵机的转动角度，返回机械臂到达的目标位置姿态)
    joint_value: The rotation angle of each servo in a list [joint1, joint2, joint3, joint4, and joint5] with units of pulse width(每个舵机转动的角度，列表形式[joint1, joint2, joint3, joint4, joint5]，单位脉宽)
    return: The 3D coordinates and posture of the target position in the format geometry_msgs/Pose(目标位置的3D坐标和位姿，格式geometry_msgs/Pose)
    '''
    msg = SetJointValue.Request()
    msg.joint_value = [float(i) for i in joint_value]
    return msg
    
if __name__ == "__main__":
    import time
    import rclpy
    from rclpy.node import Node
    import kinematics.transform as transform
    # Initialize node(初始化节点)
    rclpy.init()
    client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
    while True:
        t = time.time()
        res = node.set_pose_target([transform.link3 + transform.tool_link, 0.0, 0.36], 0.0, [-180.0, 180.0], 1.0)
        print(time.time() - t)
    rclpy.logging.get_logger('p2').info(str(res[1]))
    # print('ik', res)
    # if res[1] != []:
        # res = set_joint_value_target(res[1])
        # print('fk', res)
    node.destroy_node()
    rclpy.shutdown()
