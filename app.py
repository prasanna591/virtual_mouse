import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Hand tracking configurations
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size for PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Variables for drag-and-drop
dragging = False

# Function to calculate the distance between two landmarks
def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)

# Main loop
while True:
    success, img = cap.read()  # Read the webcam feed
    if not success:
        break

    # Flip the image horizontally for a natural selfie-view display
    img = cv2.flip(img, 1)

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the webcam image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip (landmark 8), thumb tip (landmark 4), and middle finger tip (landmark 12)
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_finger_tip = hand_landmarks.landmark[12]

            # Convert normalized hand landmark coordinates to screen coordinates
            index_x = int(index_finger_tip.x * screen_width)
            index_y = int(index_finger_tip.y * screen_height)

            # Move the mouse cursor based on the index finger tip position
            pyautogui.moveTo(index_x, index_y)

            # Calculate distances for gestures
            pinch_distance = calculate_distance(index_finger_tip, thumb_tip)
            right_click_distance = calculate_distance(middle_finger_tip, thumb_tip)

            # Pinch gesture (index and thumb close) for left-click
            if pinch_distance < 0.05:
                pyautogui.click()

            # Pinch gesture for dragging
            if pinch_distance < 0.05:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Right-click gesture (middle finger and thumb close)
            if right_click_distance < 0.05:
                pyautogui.rightClick()

            # Scroll gesture (index finger up/down while pinching)
            if dragging:
                pyautogui.scroll(int((thumb_tip.y - index_finger_tip.y) * 1000))  # Adjust sensitivity here

    # Display the webcam feed with landmarks
    cv2.imshow("Virtual Mouse", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
