import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from tf2_ros import TransformListener
from tf2_geometry_msgs import do_transform_pose
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32,Bool,Int16,UInt16
from geometry_msgs.msg import PoseStamped
class TfListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener_node')
        # 创建一个 TF 2 缓存和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        
        self.pub_robot2_pose = self.create_publisher(PoseStamped, "/robot2/goal_pose", 10)
        
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = "map"
        
        # 创建一个定时器（10Hz），每次获取一次变换
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.rotate_angle = 0
        self.error = 0.0
        self.target_angle = 360.0
        self.compute_flag = False
        self.delta_angle = 0.0 
        self.turn_angle = 0.0
        self.first_angle = 0.0
        self.get_point()


    		
        

    def timer_callback(self):
        self.get_point()


    def get_point(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'point2', rclpy.time.Time())
            quaternion = [0, 0, transform.transform.rotation.z, transform.transform.rotation.w]
            rotation = R.from_quat(quaternion)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            print("quaternion: ",quaternion)
            print("transform: ",transform.transform.translation)
            print("euler_angles: ",euler_angles)
            print("----------------------")
            self.goal_pose.pose.position.x = transform.transform.translation.x
            self.goal_pose.pose.position.y = transform.transform.translation.y
            self.goal_pose.pose.orientation.z = transform.transform.rotation.z
            self.goal_pose.pose.orientation.w = transform.transform.rotation.w
            self.pub_robot2_pose.publish(self.goal_pose)
        except (tf2_ros.TransformException, KeyError) as e:
            self.get_logger().warn(f"Could not transform: {e}")


    def normalize_angle(self,angle):
        res = angle
        #print("res: ",res)
        while res > 180:
            res -= 2.0 * 180
        while res < -180:
            res += 2.0 *180
        return res	
def main(args=None):
    rclpy.init(args=args)
    node = TfListenerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

