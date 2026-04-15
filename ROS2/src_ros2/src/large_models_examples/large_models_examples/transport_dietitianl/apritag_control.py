import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool # 导入标准服务类型
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class AprilTagHighAngleAligner(Node):
    def __init__(self):
        super().__init__('apriltag_high_angle_aligner')
        self.bridge = CvBridge()

        # ================== 状态控制变量 ==================
        self.is_align_active = False  # 默认关闭校准功能
        self.status = "idle"

        # ================== 核心参数 ==================
        self.tag_size = 0.035
        self.target_dist = 0.10
        self.tilt_angle = np.radians(60) 
        
        self.count_stop = 0
        self.smooth_x = 0.0
        self.smooth_dist = 0.0
        self.smooth_yaw = 0.0
        self.last_twist = Twist()

        # ================== 服务定义 ==================
        # 服务1: 校准开关 (true=start, false=stop)
        self.align_srv = self.create_service(SetBool, 'align_control', self.align_control_callback)
        # 服务2: 机械臂控制 (true=pick, false=place)
        self.arm_srv = self.create_service(SetBool, 'arm_control', self.arm_control_callback)

        # ================== 调试窗口 ==================
        cv2.namedWindow("High Angle Debug")
        cv2.createTrackbar("KP_Pos", "High Angle Debug", 12, 40, lambda x: None)
        cv2.createTrackbar("KP_Yaw", "High Angle Debug", 5, 20, lambda x: None)
        cv2.createTrackbar("Smooth(x0.1)", "High Angle Debug", 3, 10, lambda x: None)

        self.cmd_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.create_subscription(CameraInfo, '/depth_cam/rgb0/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/depth_cam/rgb0/image_raw', self.image_cb, 10)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        s = self.tag_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0], 
            [ s,  s, 0], 
            [ s, -s, 0], 
            [-s, -s, 0]  
        ], dtype=np.float32)

    # ================== 服务回调函数 ==================
    def align_control_callback(self, request, response):
        if request.data:
            self.is_align_active = True
            self.status = "searching"
            response.message = "Alignment Started"
        else:
            self.is_align_active = False
            self.status = "idle"
            # 停止小车运动
            self.stop_robot()
            response.message = "Alignment Stopped & Robot Braked"
        response.success = True
        self.get_logger().info(response.message)
        return response

    def arm_control_callback(self, request, response):
        if request.data:
            msg = self.pick_action()
            response.message = msg
        else:
            msg = self.place_action()
            response.message = msg
        response.success = True
        return response

    # ================== 机械臂具体动作 ==================
    def pick_action(self):
        self.get_logger().info("Executing PICK action...")
        # 在这里添加控制机械臂底层的代码
        return "Pick sequence finished"

    def place_action(self):
        self.get_logger().info("Executing PLACE action...")
        # 在这里添加控制机械臂底层的代码
        return "Place sequence finished"

    def stop_robot(self):
        empty_twist = Twist()
        self.cmd_pub.publish(empty_twist)
        self.last_twist = empty_twist

    # ================== 核心逻辑 ==================
    def normalize_angle(self, angle):
        while angle > math.pi/2: angle -= math.pi
        while angle < -math.pi/2: angle += math.pi
        return angle

    def info_cb(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)

    def image_cb(self, msg):
        if not hasattr(self, 'camera_matrix'): return
        
        kp_pos = cv2.getTrackbarPos("KP_Pos", "High Angle Debug") / 10.0
        kp_yaw = cv2.getTrackbarPos("KP_Yaw", "High Angle Debug") / 10.0
        alpha = cv2.getTrackbarPos("Smooth(x0.1)", "High Angle Debug") / 10.0
        if alpha == 0: alpha = 0.1

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        current_twist = Twist()
        display_dist_err = 0.0

        # 只有在校准激活时才计算运动逻辑
        if self.is_align_active:
            if ids is not None:
                success, rvec, tvec = cv2.solvePnP(self.obj_points, corners[0][0], self.camera_matrix, self.dist_coeffs)
                if success:
                    tx, ty, tz = tvec.flatten()
                    raw_ground_dist = tz * math.cos(self.tilt_angle) - ty * math.sin(self.tilt_angle)
                    rmat, _ = cv2.Rodrigues(rvec)
                    raw_yaw = self.normalize_angle(math.atan2(rmat[0, 2], rmat[2, 2]))

                    self.smooth_dist = self.smooth_dist * (1 - alpha) + raw_ground_dist * alpha
                    self.smooth_x = self.smooth_x * (1 - alpha) + tx * alpha
                    self.smooth_yaw = self.smooth_yaw * (1 - alpha) + raw_yaw * alpha

                    dist_err = self.smooth_dist - self.target_dist
                    display_dist_err = dist_err

                    d_pos, d_yaw = 0.025, np.radians(3.0) 
                    mode_scale = 0.4 if abs(dist_err) < 0.06 else 1.0
                    vx, vy, vw = 0.0, 0.0, 0.0

                    if abs(dist_err) > d_pos:
                        vx = float(np.clip(dist_err * kp_pos * mode_scale, -0.2, 0.2))
                    if abs(self.smooth_x) > d_pos:
                        vy = float(np.clip(-self.smooth_x * kp_pos * mode_scale, -0.2, 0.2))
                    if abs(self.smooth_yaw) > d_yaw:
                        vw = -float(np.clip(self.smooth_yaw * kp_yaw, -0.5, 0.5))

                    if abs(dist_err) <= d_pos and abs(self.smooth_x) <= d_pos and abs(self.smooth_yaw) <= d_yaw:
                        self.count_stop += 1
                        if self.count_stop >= 20: self.status = "finish"
                    else:
                        self.status = "moving"
                        self.count_stop = 0

                    current_twist.linear.x, current_twist.linear.y, current_twist.angular.z = vx, vy, vw
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.04)
            else:
                self.status = "searching"

            # 发布速度指令
            v_lpf = 0.6 
            self.last_twist.linear.x = self.last_twist.linear.x * (1-v_lpf) + current_twist.linear.x * v_lpf
            self.last_twist.linear.y = self.last_twist.linear.y * (1-v_lpf) + current_twist.linear.y * v_lpf
            self.last_twist.angular.z = self.last_twist.angular.z * (1-v_lpf) + current_twist.angular.z * v_lpf
            self.cmd_pub.publish(self.last_twist)

        # UI 绘制
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_color = (0, 255, 0) if self.status == "finish" else (0, 255, 255)
        active_str = "ACTIVE" if self.is_align_active else "DISABLED"
        cv2.putText(frame, f"ALIGN: {active_str} | {self.status.upper()}", (15, 30), font, 0.7, txt_color, 2)
        cv2.putText(frame, f"D_Err: {display_dist_err:.3f}m  X_Err: {self.smooth_x:.3f}m", (15, 60), font, 0.5, (255,255,255), 1)
        
        cmd_info = f"CMD: vx:{self.last_twist.linear.x:.2f} vy:{self.last_twist.linear.y:.2f} w:{self.last_twist.angular.z:.2f}"
        cv2.rectangle(frame, (0, frame.shape[0]-30), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        cv2.putText(frame, cmd_info, (10, frame.shape[0]-10), font, 0.5, (0, 255, 0), 1)

        cv2.imshow("High Angle Debug", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = AprilTagHighAngleAligner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()