import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.losses import BinaryCrossentropy
from model import make_embedding, make_siamese_model
from dataloader import data  # Ensure `data` is preprocessed correctly

# Load the Siamese model
embedding = make_embedding()
siamese_model = make_siamese_model(embedding)

# Define loss and optimizer
binary_cross_loss = BinaryCrossentropy()
opt = Adam(1e-4)

# Define checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Define train step function
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = [tf.expand_dims(batch[0], 0), tf.expand_dims(batch[1], 0)]  # Add batch dimension
        y = tf.expand_dims(batch[2], 0)  # Add batch dimension for the label
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss

# Define training loop
def train(data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print(f'\n Epoch {epoch}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(data))
        r = Recall()
        p = Precision()
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            progbar.update(idx + 1)
        print(f"Loss: {loss.numpy():.4f}, Recall: {r.result().numpy():.4f}, Precision: {p.result().numpy():.4f}")
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# Main execution
if __name__ == "__main__":
    # Ensure `data` is properly preprocessed as (anchor, positive/negative, label)
    EPOCHS = 50  # Set the number of epochs
    train(data, EPOCHS)
    import tensorflow as tf
    from model import make_siamese_model, make_embedding, L1Dist  # Ensure these are defined in `model.py`


    def save_model(siamese_model, file_path='siamesemodelv2.h5'):
        """
        Save the Siamese model to the specified file path.
        Args:
            siamese_model (tf.keras.Model): The trained Siamese model.
            file_path (str): Path where the model will be saved.
        """
        try:
            siamese_model.save(file_path)
            print(f"Model saved successfully at '{file_path}'")
        except Exception as e:
            print(f"Error saving model: {e}")


    def load_model(file_path='siamesemodelv2.h5'):
        """
        Load the Siamese model from the specified file path.
            file_path (str): Path to the saved model file.tf.keras.Model: Reloaded Siamese model.
        """
        try:
            siamese_model = tf.keras.models.load_model(
                file_path,
                custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
            )
            # Recompile the model if necessary
            siamese_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=tf.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            print(f"Model loaded and compiled successfully from '{file_path}'")
            return siamese_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


    def test_model(siamese_model):
        """
        Test the Siamese model with dummy data.siamese_model (tf.keras.Model): The reloaded Siamese model
        """
        # Dummy test inputs
        test_input = tf.random.normal([1, 100, 100, 3])  # Example anchor image
        test_val = tf.random.normal([1, 100, 100, 3])  # Example validation image

        try:
            predictions = siamese_model.predict([test_input, test_val])
            print("Predictions:")
            print(predictions)
        except Exception as e:
            print(f"Error testing model: {e}")


    if __name__ == "__main__":
        # Example usage

        # Step 1: Create a Siamese model instance
        embedding = make_embedding()  # Ensure make_embedding is implemented in model.py
        siamese_model = make_siamese_model(embedding)

        # Step 2: Save the model
        save_model(siamese_model, file_path='siamesemodelv2.h5')

        # Step 3: Load the model
        reloaded_model = load_model(file_path='siamesemodelv2.h5')

        # Step 4: Test the reloaded model
        if reloaded_model:
            test_model(reloaded_model)

