import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Initialize Mediapipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize K-Nearest Neighbors classifier
gesture_classifier = KNeighborsClassifier(n_neighbors=5)

# Define gesture labels (you can extend this list with your own gestures)
GESTURE_LABELS = {0: 'Fist', 1: 'Open Palm', 2: 'Peace', 3: 'Thumbs Up'}

# Function to extract hand landmarks from Mediapipe results
def extract_landmarks(hand_landmarks):
    # Extract the (x, y, z) coordinates of each landmark
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten()  # Flatten to use as input for the classifier

# Prepare training data (This is dummy data, you will need to replace it with real samples)
X_train = []
y_train = []

# Add gesture samples for training (Fist, Open Palm, etc.)
# You need to manually collect data samples for each gesture.
# Example:
# X_train.append(extract_landmarks(hand_landmarks)) -> Real sample data
# y_train.append(gesture_id) -> Label the gesture

# Train the classifier with the dataset
gesture_classifier.fit(X_train, y_train)

# Initialize the camera feed
cap = cv2.VideoCapture(0)

# Use Mediapipe Hands for hand detection and tracking
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR image to RGB and process it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # If hand landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract the landmarks for gesture recognition
                landmarks = extract_landmarks(hand_landmarks)

                # Predict the gesture
                prediction = gesture_classifier.predict([landmarks])[0]

                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the predicted gesture label
                cv2.putText(frame, GESTURE_LABELS[prediction], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the video feed
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
