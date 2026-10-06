import time
import cv2

# Configuration
flip_video = True

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Resolution & framerate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# Disable autofocus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 0)

# Allow the sensor to initialize and flush initial frames
time.sleep(0.5)
for _ in range(5):
    cap.read()

# Lock Auto White Balance and set manual color temperature ONCE
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)

# Check if the camera hardware accepted the settings
print(f"Auto WB status: {cap.get(cv2.CAP_PROP_AUTO_WB)}")
print(f"Color Temp: {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
print(f"Video flip enabled: {flip_video}")

print("Streaming active. Press 'q' in the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Rotate 180 degrees if enabled
    if flip_video:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Display feed
    cv2.imshow("Overhead Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
