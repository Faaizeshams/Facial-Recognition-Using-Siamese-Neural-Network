import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define Paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Load Image Files
anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, '*.jpg')).take(300)
positive = tf.data.Dataset.list_files(os.path.join(POS_PATH, '*.jpg')).take(300)
negative = tf.data.Dataset.list_files(os.path.join(NEG_PATH, '*.jpg')).take(300)

# Preprocess function
def preprocess(file_path):

    # Read the image
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    # Resize the image to 100x100
    img = tf.image.resize(img, (100, 100))

    # Normalize the image to be between 0 and 1
    img = img / 255.0

    return img

# Apply the preprocess function
anchor = anchor.map(preprocess)
positive = positive.map(preprocess)
negative = negative.map(preprocess)

# Create labels: 1 for positive pairs, 0 for negative pairs
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(3000))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(3000))))

# Combine the datasets
data = positives.concatenate(negatives)

# Shuffle, Cache, and Prefetch the data for better performance
data = data.shuffle(buffer_size=10000)
data = data.cache()
data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# The dataset is now ready for training or further processing.