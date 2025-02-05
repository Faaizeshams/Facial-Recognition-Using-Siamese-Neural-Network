Siamese Network for Facial Recognition
This repository contains the implementation of a Siamese Network for image verification and facial recognition. It uses TensorFlow and OpenCV to train a model that can compare pairs of images to determine if they are of the same person. The model is designed to work with facial images, but the approach can be adapted to other domains requiring image similarity comparison.

Key Features:
Siamese Network Architecture: A twin neural network architecture that learns to differentiate between pairs of images by computing a similarity score.
Image Collection: Real-time image collection using a webcam. Users can collect anchor, positive, and negative images by pressing specific keys.
Data Preprocessing: Images are preprocessed (resized and normalized) to prepare them for training.
Training and Evaluation: The network is trained on the Labeled Faces in the Wild (LFW) dataset and evaluated using Precision and Recall metrics.
Real-Time Face Verification: The trained model can be used for real-time face verification using a webcam.
Model Checkpoints: The model is saved at regular intervals during training, ensuring recovery and model persistence.
Thresholding for Prediction: A threshold-based system is used to determine whether a given pair of images are from the same person or not.
Prerequisites:
Python 3.x
TensorFlow
OpenCV
Matplotlib
NumPy
This project uses the Labeled Faces in the Wild (LFW) dataset for training. The dataset can be downloaded from LFW Website.

How to Use:
Setup and Dataset Extraction: Run setup_and_extract_dataset() to extract and organize the LFW dataset.
Image Collection: Run the collect_images() function to collect real-time images through the webcam (Press 'a' for anchor images, 'p' for positive images, and 'q' to quit).
Train the Model: Use the train() function to train the Siamese network.
Face Verification: Once trained, use the webcam to perform face verification in real-time by pressing 'v' to save input images and trigger verification.
Training:
The model uses binary cross-entropy loss and Adam optimizer for training.
The training loop saves checkpoints every 10 epochs.
The model architecture includes convolutional layers for feature extraction and a custom L1 distance layer to compute the similarity between image pairs.
Real-Time Testing:
The model allows for live testing using OpenCV. It captures a frame from the webcam, performs image verification with a stored input image, and prints the verification result.

Model Saving and Loading:
The trained model can be saved to disk using model.save() and reloaded for later predictions.
Files:
main.py: Contains the full implementation of the Siamese network, image collection, training, and verification pipeline.
requirements.txt: List of required Python packages.
training_checkpoints/: Folder for saving model checkpoints during training.
application_data/: Folder containing the input image and verification images for real-time testing.
