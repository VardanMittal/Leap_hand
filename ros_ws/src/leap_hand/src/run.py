#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np

class LeapHandPublisher(Node):
    def __init__(self):
        super().__init__('leap_hand_publisher')
        
        # Publisher for LEAP Hand joint states
        self.publisher_ = self.create_publisher(JointState, '/leaphand_node/cmd_leap', 10)
        
        # Subscriber to receive gestures from GestureClassifierNode
        self.create_subscription(String, 'human_gesture', self.gesture_callback, 10)

        # Initialize JointState message
        self.joint_state = JointState()
        self.joint_state.name = [f'joint_{i}' for i in range(16)]  # Assuming 16 joints
        
        # Define gesture positions (example values)
        self.gesture_positions = {
        "rock": np.array([np.radians(180), np.radians(240), np.radians(261), np.radians(253),
                          np.radians(180), np.radians(236), np.radians(295), np.radians(243),
                          np.radians(180), np.radians(242), np.radians(270), np.radians(255),
                          np.radians(149), np.radians(87), np.radians(268), np.radians(253)]),
       
        "paper": np.array([np.radians(180)] * 16),  # All joints flat for open fingers

        "scissors": np.array([np.radians(180), np.radians(180), np.radians(180),
                              np.radians(180), np.radians(180), np.radians(180),
                              np.radians(180), np.radians(180), np.radians(180),
                              np.radians(242), np.radians(270), np.radians(255),
                              np.radians(149), np.radians(87), np.radians(268),
                              np.radians(253)]),
    }

    def gesture_callback(self, msg):
        gesture = msg.data
        self.get_logger().info(f"Received gesture: {gesture}")

        # Set joint positions based on the detected gesture
        if gesture == "rock":
            self.joint_state.position = self.gesture_positions["paper"]
        elif gesture == "paper":
            self.joint_state.position = self.gesture_positions["scissors"]
        elif gesture == "scissors":
            self.joint_state.position = self.gesture_positions["rock"]

        # Publish the joint state to the LEAP Hand
        self.publisher_.publish(self.joint_state)
        self.get_logger().info(f"Published positions for gesture '{gesture}': {self.joint_state.position}")

def main(args=None):
    rclpy.init(args=args)
    leap_hand_publisher = LeapHandPublisher()
    rclpy.spin(leap_hand_publisher)
    leap_hand_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
