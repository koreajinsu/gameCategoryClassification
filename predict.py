import tensorflow as tf
import numpy as np

# Load the test data
X_test = np.load('./datasets/news_data_X_test.npy')
Y_test = np.load('./datasets/news_data_Y_test.npy')

# Load the saved model from the .h5 file
model = tf.keras.models.load_model('./models/final_model.h5')  # Use the path to your .h5 file

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)

# Print the results
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
