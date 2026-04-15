#!/usr/bin/python3
# coding=utf8
# @Author: Aiden
# @Date: 2024/12/31
import os
import time
import rclpy
import numpy as np
from kinematics.kinematics_control import set_pose_target
from servo_controller.bus_servo_control import set_servo_position

stop = False
chassis_type = os.environ['MACHINE_TYPE']

def interrupt(status=False):
    global stop
    stop = status

def send_request(client, msg):
    future = client.call_async(msg)
    while rclpy.ok():
        if future.done() and future.result():
            return future.result()

dt = 0.1
d = 0.015
def pick_without_back(position, pitch, yaw, gripper_angle, gripper_depth, joints_pub, kinematics_client, interpolation=False):
    global stop
    if not stop:
        position[2] += 0.01
        
        if interpolation:
            msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=0.1)
            res = send_request(kinematics_client, msg)
            servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
            set_servo_position(joints_pub, 1.0, ((1, servo_data[-1][0]),))
            
            msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=1.5)
            res = send_request(kinematics_client, msg)
            servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
            for i in servo_data:
                set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                time.sleep(dt - d)
        else:
            msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0)
            res = send_request(kinematics_client, msg)
            servo_data = res.pulse
            set_servo_position(joints_pub, 1.0, ((1, servo_data[0]),))
            time.sleep(1)

            set_servo_position(joints_pub, 1, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)

        set_servo_position(joints_pub, 0.5, ((5, yaw),))
        time.sleep(0.5)
        
        if not stop:
            position[2] -= (0.02 + gripper_depth)   #position[2] -= (0.03 + gripper_depth)   修改夹取深度
            
            if interpolation:
                msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=0.8)
                res = send_request(kinematics_client, msg)
                servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
                for i in servo_data:
                    set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                    time.sleep(dt - d)
            else:
                msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0)
                res = send_request(kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(joints_pub, 0.5, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(0.5)

            set_servo_position(joints_pub, 0.5, ((10, gripper_angle),))
            time.sleep(0.5)

            if not stop:
                position[2] += (0.02 + gripper_depth)

                if interpolation:
                    msg = set_pose_target(position, pitch, [-180.0, 180.0], 1.0, duration=0.8)
                    res = send_request(kinematics_client, msg)
                    servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
                    for i in servo_data:
                        set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                        time.sleep(dt - d)
                else:
                    msg = set_pose_target(position, pitch, [-180.0, 180.0], 1.0)
                    res = send_request(kinematics_client, msg)
                    servo_data = res.pulse
                    set_servo_position(joints_pub, 0.5, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                    time.sleep(0.5)
                return True
    return False


def pick(position, pitch, yaw, gripper_angle, gripper_depth, joints_pub, kinematics_client, interpolation=False):
    global stop
    if pick_without_back(position, pitch, yaw, gripper_angle, gripper_depth, joints_pub, kinematics_client, interpolation):
        if not stop:
            if interpolation:
                if chassis_type == 'Slide_Rails':
                    msg = set_pose_target([0.11, 0.0, 0.15], 73, [-180.0, 180.0], 1.0, duration=1.0)
                else:
                    msg = set_pose_target([0.11, 0.0, 0.09], 73, [-180.0, 180.0], 1.0, duration=1.0)
                res = send_request(kinematics_client, msg)
                servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
                set_servo_position(joints_pub, 0.5, ((5, 500),))
                time.sleep(0.02)
                for i in servo_data:
                    set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                    time.sleep(dt - d)
            else:
                if chassis_type == 'Slide_Rails':
                    msg = set_pose_target([0.11, 0.0, 0.15], 73, [-180.0, 180.0], 1.0)
                else:
                    msg = set_pose_target([0.11, 0.0, 0.09], 73, [-180.0, 180.0], 1.0)
                res = send_request(kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(joints_pub, 0.5, ((5, 500),))
                time.sleep(0.02)
                set_servo_position(joints_pub, 1, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)
            return True
    return False

def place(position, pitch, yaw, gripper_angle, joints_pub, kinematics_client, interpolation=False):
    global stop
    if not stop:
        position[2] += 0.03
        
        if interpolation:
            msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=0.1)
            res = send_request(kinematics_client, msg)
            servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
            set_servo_position(joints_pub, 1.0, ((1, servo_data[-1][0]),))
            time.sleep(1)

            msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=2.0)
            res = send_request(kinematics_client, msg)
            servo_data1 = np.array(res.pulse).reshape(-1, 5).tolist()
            for i in servo_data1:
                set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                time.sleep(dt - d)
        else:
            msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0)
            res = send_request(kinematics_client, msg)
            servo_data = res.pulse
            set_servo_position(joints_pub, 1.0, ((1, servo_data[0]),))
            time.sleep(1)
            print("11")
            # self.get_logger().info(f'222:{position}')
            set_servo_position(joints_pub, 1, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            print("22")
            # self.get_logger().info(f'333:{position}')
            time.sleep(1)

        if not stop:
            position[2] -= 0.03
            
            set_servo_position(joints_pub, 0.5, ((5, yaw),))
            time.sleep(0.8)
            
            if interpolation:
                msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=0.8)
                res = send_request(kinematics_client, msg)
                servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
                for i in servo_data:
                    set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                    time.sleep(dt - d)
            else:
                msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0)
                res = send_request(kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(joints_pub, 1, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1.2)

            set_servo_position(joints_pub, 0.5, ((10, gripper_angle),))
            time.sleep(0.5)
            
            position[2] += 0.03
            if interpolation:
                msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0, duration=0.8)
                res = send_request(kinematics_client, msg)
                servo_data = np.array(res.pulse).reshape(-1, 5).tolist()
                for i in servo_data:
                    set_servo_position(joints_pub, dt, ((2, i[1]), (3, i[2]), (4, i[3])))
                    time.sleep(dt - d)          
            else:
                msg = set_pose_target(position, pitch, [-90.0, 90.0], 1.0)
                res = send_request(kinematics_client, msg)
                servo_data = res.pulse
                set_servo_position(joints_pub, 0.5, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(0.5)
            set_servo_position(joints_pub, 0.5, ((10, 200),))
            return True
    return False

