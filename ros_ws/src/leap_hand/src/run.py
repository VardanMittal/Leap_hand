#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np

class LeapHandPublisher:
    def __init__(self):
        rospy.init_node('leap_hand_publisher', anonymous=True)
        
        # Publisher for LEAP Hand joint states
        self.publisher_ = rospy.Publisher('/leaphand_node/cmd_leap', JointState, queue_size=10)
        
        # Subscriber to receive gestures from GestureClassifierNode
        rospy.Subscriber('human_gesture', String, self.gesture_callback)

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
        rospy.loginfo(f"Received gesture: {gesture}")

        # Set joint positions based on the detected gesture
        if gesture == "rock":
            self.joint_state.position = self.gesture_positions["rock"]
        elif gesture == "paper":
            self.joint_state.position = self.gesture_positions["paper"]
        elif gesture == "scissors":
            self.joint_state.position = self.gesture_positions["scissors"]

        # Publish the joint state to the LEAP Hand
        self.publisher_.publish(self.joint_state)
        rospy.loginfo(f"Published positions for gesture '{gesture}': {self.joint_state.position}")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    leap_hand_publisher = LeapHandPublisher()
    leap_hand_publisher.spin()
