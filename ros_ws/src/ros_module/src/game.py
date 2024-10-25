#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np

class AlwaysWinNode:
    def __init__(self):
        rospy.init_node("always_win_node")
        
        # Define mappings for gestures to LEAP Hand poses (replace with actual poses)
        self.gesture_poses = {
            "rock": np.array([0.0] * 16),      # Replace with hand pose for "rock"
            "paper": np.array([0.5] * 16),     # Replace with hand pose for "paper"
            "scissors": np.array([1.0] * 16),  # Replace with hand pose for "scissors"
        }
        
        # Subscriber to opponent's gesture
        rospy.Subscriber("/robot_gesture", String, self.gesture_callback)
        
        # Publisher to LEAP Hand command topic
        self.leap_pub = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=10)
        
        rospy.loginfo("AlwaysWinNode initialized.")

    def gesture_callback(self, msg):
        # Interpret opponent's gesture and select winning response
        opponent_gesture = msg.data
        rospy.loginfo(f"Opponent gesture: {opponent_gesture}")
        
        # Determine winning gesture
        winning_gesture = self.get_winning_gesture(opponent_gesture)
        
        # Publish corresponding LEAP Hand command
        self.publish_leap_gesture(winning_gesture)

    def get_winning_gesture(self, opponent_gesture):
        # Define winning responses
        if opponent_gesture == "rock":
            return "paper"
        elif opponent_gesture == "paper":
            return "scissors"
        elif opponent_gesture == "scissors":
            return "rock"
        else:
            rospy.logwarn("Unknown gesture received!")
            return None
        
    def publish_leap_gesture(self, gesture):
        if gesture in self.gesture_poses:
            joint_state_msg = JointState()
            joint_state_msg.position = self.gesture_poses[gesture]
            self.leap_pub.publish(joint_state_msg)
            rospy.loginfo(f"Responding with gesture: {gesture}")
        else:
            rospy.logwarn("No valid gesture to respond with.")

def main():
    node = AlwaysWinNode()
    rospy.spin()

if __name__ == "__main__":
    main()
