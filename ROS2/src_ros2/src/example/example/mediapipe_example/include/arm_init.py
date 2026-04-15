#!/usr/bin/env python3
# encoding: utf-8
from servo_controller_msgs.msg import ServoPosition, ServosPosition
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
import time

class ServoControlNode(Node):
    def __init__(self, node_name='servo_control'):
        super().__init__(node_name)
        
        self.publisher = self.create_publisher(
            ServosPosition, 
            'servo_controller', 
            1
        )
        namespace = self.get_namespace()
        if namespace == '/':
            namespace = ''

        self.client = self.create_client(
            Trigger, 
            namespace + '/controller_manager/init_finish'
        )
        

        self.get_logger().info("Waiting for init_finish service...")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
            
        
        self.execute_servo_sequence()

    def set_servo_position(self, duration, positions):
        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position = [
            self.create_servo_position(id, pos) for id, pos in positions
        ]
        msg.position_unit = "pulse"
        self.publisher.publish(msg)
        self.get_logger().info(f"Published servo positions: {positions}")

    @staticmethod
    def create_servo_position(servo_id, position):
        pos = ServoPosition()
        pos.id = servo_id
        pos.position = float(position)
        return pos

    def execute_servo_sequence(self):
        try:
           
            self.set_servo_position(
                duration=2,
                positions=(
                    (10, 500), 
                    (5, 500),
                    (4, 220),
                    (3, 210),
                    (2, 720),
                    (1, 500)
                )
            )
            time.sleep(3)  
        except Exception as e:
            self.get_logger().error(f"Action failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ServoControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
