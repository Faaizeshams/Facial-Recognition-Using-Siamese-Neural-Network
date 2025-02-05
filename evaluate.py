import tensorflow as tf
from model import L1Dist

# Function to load the saved model
def load_model(model_path):
    if tf.io.gfile.exists(model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
        )
        print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

if __name__ == "__main__":
    model_path = 'siamesemodelv2.h5'
    siamese_model = load_model(model_path)

    # Test data (replace with your actual data)
    test_input = tf.random.normal((1, 100, 100, 3))
    test_val = tf.random.normal((1, 100, 100, 3))

    prediction = siamese_model.predict([test_input, test_val])
    print("Prediction:", prediction)
    siamese_model.summary()
