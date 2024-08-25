import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Gesture Recognizer
# (If 'mp.tasks' is unavailable, use alternative gesture recognition)
model_path = 'gesture_recognizer.task'  # Ensure this path is correct

# Define a simple gesture recognizer (this should be replaced with actual model predictions)
import math

def calculate_finger_tip_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

import math

def calculate_finger_tip_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

import math

def calculate_finger_tip_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c) with b as the vertex"""
    ab = (b.x - a.x, b.y - a.y, b.z - a.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot_product = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    angle = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    return math.degrees(angle)

import math

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3

    vec1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    vec2 = np.array([x3 - x2, y3 - y2, z3 - z2])
    
    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate the magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (norm1 * norm2))
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Function to convert hand landmarks to a numpy array
def hand_landmarks_to_numpy_array(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

# Function to recognize hand gestures based on landmarks
def recognize_gesture(hand_landmarks):
    # Convert hand landmarks to numpy array
    landmarks = hand_landmarks_to_numpy_array(hand_landmarks)

    # Calculate distances between finger tips
    thumb_index_distance = calculate_distance(landmarks[4], landmarks[8])
    index_middle_distance = calculate_distance(landmarks[8], landmarks[12])
    middle_ring_distance = calculate_distance(landmarks[12], landmarks[16])
    ring_pinky_distance = calculate_distance(landmarks[16], landmarks[20])

    # Calculate distances between base of fingers and tips
    thumb_length = calculate_distance(landmarks[2], landmarks[4])
    index_length = calculate_distance(landmarks[5], landmarks[8])
    middle_length = calculate_distance(landmarks[9], landmarks[12])
    ring_length = calculate_distance(landmarks[13], landmarks[16])
    pinky_length = calculate_distance(landmarks[17], landmarks[20])

    # Thresholds for lengths and distances
    length_threshold_open = 0.2  # Adjust these thresholds based on your scale
    length_threshold_closed = 0.1
    distance_threshold_thumb_index = 0.1

    # Determine finger states
    thumb_is_open = thumb_length > length_threshold_open
    index_is_open = index_length > length_threshold_open
    middle_is_open = middle_length > length_threshold_open
    ring_is_open = ring_length > length_threshold_open
    pinky_is_open = pinky_length > length_threshold_open

    thumb_is_closed = thumb_length < length_threshold_closed
    index_is_closed = index_length < length_threshold_closed
    middle_is_closed = middle_length < length_threshold_closed
    ring_is_closed = ring_length < length_threshold_closed
    pinky_is_closed = pinky_length < length_threshold_closed

    if all([thumb_is_open, index_is_open, middle_is_open, ring_is_open, pinky_is_open]):
        return "Open Hand"
    elif all([thumb_is_closed, index_is_closed, middle_is_closed, ring_is_closed, pinky_is_closed]):
        return "Fist"
    elif index_is_open and middle_is_open and ring_is_closed and pinky_is_closed and thumb_is_closed:
        return "Peace"
    elif thumb_is_open and index_is_closed and middle_is_closed and ring_is_closed and pinky_is_closed and thumb_index_distance > distance_threshold_thumb_index:
        return "Thumbs Up"
    elif thumb_is_open and index_is_open and middle_is_closed and ring_is_closed and pinky_is_closed and thumb_index_distance > distance_threshold_thumb_index:
        return "Call Me"
    elif index_is_open and middle_is_closed and ring_is_open and pinky_is_open and thumb_is_closed and index_middle_distance > distance_threshold_thumb_index:
        return "Rock"
    else:
        return "Unknown"



# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

def process_gesture(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            gesture = recognize_gesture(landmarks)
            return gesture
    return "Gesture not recognized"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform hand landmark detection
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        results = hands.process(rgb_frame)
        
        # Draw landmarks and connections
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Process the frame for gesture recognition
            gesture = process_gesture(frame, results)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
