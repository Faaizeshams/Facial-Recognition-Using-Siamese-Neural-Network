import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from model import make_embedding, make_siamese_model, L1Dist
from dataloader import preprocess  # Ensure this is implemented

# Paths
application_data_path = 'application_data'
verification_images_path = os.path.join(application_data_path, 'verification_images')
input_image_path = os.path.join(application_data_path, 'input_image', 'input_image.jpg')

# Load the trained model
model = tf.keras.models.load_model(
    'siamesemodelv2.h5',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
)

# Helper function to validate and re-save images
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify integrity of the image
        return True
    except Exception as e:
        print(f"Invalid image {file_path}: {e}")
        return False

def clean_verification_folder(folder_path):
    #Remove invalid or hidden files from the verification folder.
    for image in os.listdir(folder_path):
        if image.startswith('.'):
            print(f"Skipping hidden or invalid file: {image}")
            continue
        image_path = os.path.join(folder_path, image)
        if not is_valid_image(image_path):
            os.remove(image_path)
            print(f"Removed invalid image: {image}")
# Verification function
def verify(model, detection_threshold, verification_threshold):
    #Verifies the input image against stored verification images.
    results = []
    if not os.path.exists(input_image_path):
        print("Input image not found.")
        return [], False

    input_img = preprocess(input_image_path)
    for image in os.listdir(verification_images_path):
        if image.startswith('.'):
            print(f"Skipping hidden or invalid file: {image}")
            continue

        image_path = os.path.join(verification_images_path, image)
        if not is_valid_image(image_path):
            continue

        validation_img = preprocess(image_path)
        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        print(f"Prediction for {image}: {result}")  # Debugging output
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(verification_images_path))
    verified = verification > verification_threshold

    # Print detection count and verification ratio
    print(f"Detection Count: {detection}")
    print(f"Verification Ratio: {verification}")

    return results, verified

# Counter for verification attempts
verification_attempts = 0



# OpenCV Real-Time Verification
cap = cv2.VideoCapture(0)  # Use appropriate index for your webcam

detection_threshold = 0.3
verification_threshold = 0.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    size = 250
    x1, y1 = center_x - size // 2, center_y - size // 2
    x2, y2 = center_x + size // 2, center_y + size // 2
    frame = frame[y1:y2, x1:x2]

    cv2.imshow('Verification', frame)

    if cv2.waitKey(10) & 0xFF == ord('v'):
        if verification_attempts < 2:
            os.makedirs(os.path.dirname(input_image_path), exist_ok=True)
            cv2.imwrite(input_image_path, frame)
            print("Saved input image.")

            results, verified = verify(model, detection_threshold, verification_threshold)
            print("Verification Results:", results)
            #print("Verified" if verified else "Not Verified")

            if verified:
                print("\n\nPerson has been verified: VERIFIED")

            else:
                print("\n\nPerson has not been verified: NOT VERIFIED")

            verification_attempts += 1
        else:
            print("\n\n\n\nVerification attempts limit reached.")
            sys.exit()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
