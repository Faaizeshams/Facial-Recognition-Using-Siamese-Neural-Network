import cv2
import os
from matplotlib import pyplot as plt
import uuid  # Import uuid library to generate unique image names

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setup paths for datasets
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make the directories if they don't exist
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

print("Directories created successfully!")

# Start Webcam Capture and Image Saving Logic
cap = cv2.VideoCapture(0)  # Try with 0 (or adjust if needed)

if not cap.isOpened():  # Check if the webcam was successfully opened
    print("Error: Could not open webcam!")
    exit()  # Exit if webcam cannot be opened

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame!")
        break  # Exit if no frame was captured

    # Get the center of the frame
    height, width, _ = frame.shape

    # Define the new crop size (250x250 pixels)
    crop_size = 250

    # Calculate the starting point for the crop to center it
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    # Crop the frame to the new size centered around the middle of the frame
    frame_cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Wait for a key press once per loop
    key = cv2.waitKey(1) & 0xFF

    # Collect anchor images when the 'a' key is pressed
    if key == ord('a'):
        # Create a unique file path for anchor images using uuid
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Save the anchor image to the file
        cv2.imwrite(imgname, frame_cropped)

    # Collect positive images when the 'p' key is pressed
    if key == ord('p'):
        # Create a unique file path for positive images using uuid
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Save the positive image to the file
        cv2.imwrite(imgname, frame_cropped)

    # Graceful exit when the 'q' key is pressed
    if key == ord('q'):
        print("Exiting...")
        break

    # Display the captured frame
    cv2.imshow('Image Collection', frame_cropped)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Optionally, display the last captured frame (for debugging)
plt.imshow(frame_cropped)
