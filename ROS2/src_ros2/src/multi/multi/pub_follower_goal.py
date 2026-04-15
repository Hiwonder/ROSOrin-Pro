import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped,Twist,PoseWithCovarianceStamped
from nav2_msgs.action import NavigateThroughPoses
from rclpy.node import Node
from std_msgs.msg import UInt16,Bool,Int16
import time
from visualization_msgs.msg import MarkerArray

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf2_ros

from transforms3d.euler import euler2quat, quat2euler
import math
class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')
        self._client = ActionClient(self, NavigateThroughPoses, '/navigate_through_poses')
        self.get_goal_pose = self.create_subscription(PoseStamped,"/robot_1/goal_pose",self.get_GoalPoseCallBack,1)
        self.get_waypoints = self.create_subscription(MarkerArray,"/waypoints",self.get_waypointsPoseCallBack,1)
        self.pub_done = self.create_publisher(Bool, "/transport_done", 10)
        self.pub_robot_pose = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.goal_pose = PoseStamped()
        self.orinal_pose = PoseStamped()
        self.goal_pose.header.frame_id = "map"
        self.orinal_pose.header.frame_id = "map"
        
        
        self.robot_1_to_point2_broadcaster = StaticTransformBroadcaster(self)
        self.robot_1_to_point3_broadcaster = StaticTransformBroadcaster(self)
        
         
        self.dist = 0.3
        self.declare_parameter("queue_name", "convoy")
        self.queue = self.get_parameter("queue_name").get_parameter_value().string_value #row-行,convoy-护卫队,column-列
        
        self.declare_parameter("dist", 0.3)
        self.dist = self.get_parameter("dist").get_parameter_value().double_value 
        
    def get_waypointsPoseCallBack(self,msg):
        #print("get the pose: ")
        #print(len(msg.markers))
        #print(msg.markers[len(msg.markers)-2].pose)
        self.goal_pose.pose.position.x = msg.markers[len(msg.markers)-2].pose.position.x
        self.goal_pose.pose.position.y = msg.markers[len(msg.markers)-2].pose.position.y  
        self.goal_pose.pose.orientation.x = msg.markers[len(msg.markers)-2].pose.orientation.x
        self.goal_pose.pose.orientation.y = msg.markers[len(msg.markers)-2].pose.orientation.y
        self.goal_pose.pose.orientation.z = msg.markers[len(msg.markers)-2].pose.orientation.z
        self.goal_pose.pose.orientation.w = msg.markers[len(msg.markers)-2].pose.orientation.w
        self.send_goal()
            
       	

    def get_GoalPoseCallBack(self,msg):
        robot_transform = TransformStamped()
        robot3_transform = TransformStamped()
        
        robot_transform.header.stamp = self.get_clock().now().to_msg()
        robot_transform.header.frame_id = "robot_1/base_link"
        robot_transform.child_frame_id = "point2"
        
        robot3_transform.header.stamp = self.get_clock().now().to_msg()
        robot3_transform.header.frame_id = "robot_1/base_link"
        robot3_transform.child_frame_id = "point3"
               
        if self.queue == "column":
        	robot_transform.transform.translation.x =  -self.dist
        	robot_transform.transform.translation.y =  0.0
        	
        	robot3_transform.transform.translation.x =  -self.dist*2
        	robot3_transform.transform.translation.y =  0.0
        	
        elif self.queue == "row":
        	robot_transform.transform.translation.x =  0.0
        	robot_transform.transform.translation.y =  -self.dist
        	
        	robot3_transform.transform.translation.x =  0.0
        	robot3_transform.transform.translation.y =  self.dist
        	        	
        elif self.queue == "convoy":
        	robot_transform.transform.translation.x =  -self.dist
        	robot_transform.transform.translation.y =  -self.dist
        	
        	robot3_transform.transform.translation.x =  -self.dist
        	robot3_transform.transform.translation.y =  self.dist

        robot_transform.transform.rotation.w = 1.0
        robot3_transform.transform.rotation.w = 1.0
        
        self.robot_1_to_point2_broadcaster.sendTransform(robot_transform)   
        self.robot_1_to_point3_broadcaster.sendTransform(robot3_transform)
        print("send TF.")

        

          
        

    def send_goal(self):
        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses.append(self.goal_pose)
        self._client.wait_for_server()
        self._client.send_goal_async(goal_msg,feedback_callback=self.feedback_callback).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        result = future.result()

        if not result.accepted:
            self.get_logger().error('Goal was rejected!')
            return
        self.get_logger().info('Goal accepted, waiting for result...')
        result_msg = result
        #print(result_msg)
        result.get_result_async().add_done_callback(self.result_callback)
        
    def result_callback(self, future):
        result_msg = future.result()
        print("the result_status is ",result_msg.status)
        if result_msg.status==4:
            self.pub_rotate = False
            print("Goal reached.")


        
    def feedback_callback(self,feedback_msg):
        if feedback_msg.feedback.distance_remaining<0.10:
            print("Done.")



def main(args=None):
    rclpy.init(args=args)
    client = NavigationClient()
    rclpy.spin(client)

if __name__ == '__main__':
    main()

