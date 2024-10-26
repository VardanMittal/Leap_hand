#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import mediapipe as mp

class GestureClassifierNode:
    def __init__(self):
        rospy.init_node('gesture_classifier', anonymous=True)
        self.publisher_ = rospy.Publisher('human_gesture', String, queue_size=10)

        # Initialize MediaPipe Hands for hand gesture recognition
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Set a timer for gesture classification
        self.rate = rospy.Rate(10)  # 10 Hz

    def classify_gesture(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image from webcam.")
                break

            # Convert the frame to RGB and process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Default gesture is "unknown" if no hand is detected
            gesture = "unknown"

            if results.multi_hand_landmarks:
                # Detect gesture based on landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture = self.detect_gesture(hand_landmarks.landmark)

                # Draw hand landmarks on frame
                for hand_landmark in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

            # Publish the detected gesture
            msg = String()
            msg.data = gesture
            self.publisher_.publish(msg)
            rospy.loginfo(f"Detected gesture: {gesture}")

            # Display the image with the detected gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Detection", frame)
            cv2.waitKey(1)

            self.rate.sleep()

    def is_finger_closed(self, finger_tip, wrist):
        return (wrist.y - finger_tip.y < 0.45)

    def detect_gesture(self, landmarks):
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]

        if (self.is_finger_closed(index_tip, wrist) and
            self.is_finger_closed(middle_tip, wrist) and
            self.is_finger_closed(ring_tip, wrist) and
            self.is_finger_closed(pinky_tip, wrist)):
            return "rock"
        elif (not self.is_finger_closed(index_tip, wrist) and
              not self.is_finger_closed(middle_tip, wrist) and
              self.is_finger_closed(ring_tip, wrist) and
              self.is_finger_closed(pinky_tip, wrist)):
            return "scissors"
        elif (not self.is_finger_closed(index_tip, wrist) and
              not self.is_finger_closed(middle_tip, wrist) and
              not self.is_finger_closed(ring_tip, wrist) and
              not self.is_finger_closed(pinky_tip, wrist)):
            return "paper"
        else:
            return "unknown"

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        gesture_classifier = GestureClassifierNode()
        gesture_classifier.classify_gesture()
    except rospy.ROSInterruptException:
        gesture_classifier.cleanup()
