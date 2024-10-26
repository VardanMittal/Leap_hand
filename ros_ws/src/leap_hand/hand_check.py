import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(landmarks):
    # Get the positions of specific landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    # Check for gestures based on finger positions
    if thumb_tip < index_tip < middle_tip < ring_tip < pinky_tip:  # Open hand
        return "Paper"
    elif thumb_tip < index_tip and middle_tip < ring_tip < pinky_tip:  # Scissors
        return "Scissors"
    elif index_tip < middle_tip and ring_tip < pinky_tip:  # Closed fist
        return "Rock"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect gesture
                gesture = detect_gesture(hand_landmarks.landmark)
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(landmarks):
    # Get the positions of specific landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    # Check for gestures based on finger positions
    if thumb_tip < index_tip < middle_tip < ring_tip < pinky_tip:  # Open hand
        return "Paper"
    elif thumb_tip < index_tip and middle_tip < ring_tip < pinky_tip:  # Scissors
        return "Scissors"
    elif index_tip < middle_tip and ring_tip < pinky_tip:  # Closed fist
        return "Rock"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect gesture
                gesture = detect_gesture(hand_landmarks.landmark)
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
