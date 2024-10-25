#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import mediapipe as mp

class GestureClassifierNode(Node):
    def __init__(self):
        super().__init__('gesture_classifier')
        self.publisher_ = self.create_publisher(String, 'robot_gesture', 10)
        
        # Initialize MediaPipe Hands for hand gesture recognition
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Set timer for callback
        self.timer = self.create_timer(0.1, self.classify_gesture_callback)

    def classify_gesture_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image from webcam.")
            return

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
        self.get_logger().info(f"Detected gesture: {gesture}")
        
        # Display the image with the detected gesture
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Detection", frame)
        cv2.waitKey(1)

    def detect_gesture(self, landmarks):
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
       
        if (self.is_finger_closed(thumb_tip, wrist) and
            self.is_finger_closed(index_tip, wrist) and
            self.is_finger_closed(middle_tip, wrist)):
            return "rock"
        elif (self.is_finger_closed(thumb_tip, wrist) and
              not self.is_finger_closed(index_tip, wrist) and
              not self.is_finger_closed(middle_tip, wrist)):
            return "scissors"
        else:
            return "paper"

    def is_finger_closed(self, finger_tip, wrist):
        # Implement a method to determine if a finger is closed
        distance = np.linalg.norm(np.array([finger_tip.x, finger_tip.y]) - np.array([wrist.x, wrist.y]))
        return distance < 0.1  # Adjust threshold as needed

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
