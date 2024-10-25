import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class GestureDetectorNode(Node):
    def __init__(self):
        super().__init__('gesture_detector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create a publisher to publish gesture detection results (if needed)
        self.gesture_pub = self.create_publisher(Image, 'gesture_detection', 10)

        # Create a timer to process frames at a set frequency
        self.timer = self.create_timer(0.1, self.capture_and_process)

        # Initialize the video capture (webcam)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open video source.")
            return

    def capture_and_process(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Warning: Could not read frame.")
            return

        # Process the frame to detect gestures
        processed_frame = self.detect_gesture(frame)

        # Convert the OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")

        # Publish the processed frame
        self.gesture_pub.publish(ros_image)

        # Optionally, display the frame (for debugging)
        cv2.imshow('Gesture Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()

    def detect_gesture(self, frame):
        # Placeholder for gesture detection logic
        # Replace with actual gesture detection code
        # E.g., color detection, hand tracking, or ML model

        # Here we simply add text as a placeholder for a detected gesture
        cv2.putText(frame, "Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame

    def destroy_node(self):
        # Properly release the camera
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    gesture_detector_node = GestureDetectorNode()
    rclpy.spin(gesture_detector_node)

    # Cleanup when done
    gesture_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
