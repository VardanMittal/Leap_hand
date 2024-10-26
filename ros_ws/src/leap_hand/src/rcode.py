#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from leap_hand.srv import *
from std_msgs.msg import String

# This is example code, it reads the position from LEAP Hand and commands it
# Be sure to query the services only when you need it
# This way you don't clog up the communication lines and you get the newest data possible
class r:
    def __init__(self):        
        self.pub_hand = rospy.Publisher("/leaphand_node/cmd_ones", JointState, queue_size = 3) 
        rospy.Subscriber('human_gesture', String, self.gesture_callback)
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
        stater = JointState()
        
        # Set joint positions based on the detected gesture
        if gesture == "rock":
            stater.position  = self.gesture_positions["paper"]
        elif gesture == "paper":
            stater.position  = self.gesture_positions["scissors"]
        elif gesture == "scissors":
            stater.position  = self.gesture_positions["rock"]
        
        self.pub_hand.publish(stater)  ##choose the right embodiment here
        rospy.loginfo(f"Published position")

           
if __name__ == "__main__":
    rospy.init_node("publisher")
    telekinesis_node = r()
    rospy.spin()
