import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('D:/X/AI/Internships/Prodigy InfoTech/Tasks/4 Hand Gesture Recognition/gesture_recognition_model.h5')


# Define the size to which images will be resized
resize_width = 128
resize_height = 128

# List of gestures
gestures = [
"like", "dislike", "peace", "one", "fist", "Hello", "Love you"
]

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    back_sub = cv2.createBackgroundSubtractorMOG2()
    fg_mask = back_sub.apply(gray)

    # Apply mask to the original image
    fg_image = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Resize the image
    resized_img = cv2.resize(fg_image, (resize_width, resize_height))
    resized_img = resized_img.astype("float32") / 255.0
    resized_img = np.expand_dims(resized_img, axis=0)
    resized_img = np.expand_dims(resized_img, axis=-1)  # Add channel dimension if required

    return resized_img

def predict_gesture(frame):
    img = preprocess_image(frame)
    prediction = model.predict(img)
    gesture_index = np.argmax(prediction)
    return gestures[gesture_index]

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the gesture prediction
    gesture = predict_gesture(frame)
    
    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle around the largest contour (assuming it's the hand)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, gesture, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
