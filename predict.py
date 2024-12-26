import tensorflow as tf
import numpy as np

# Define the custom lambda functions used in the model
def attention_weighted_sum(x):
    import tensorflow as tf  # Explicitly import TensorFlow inside the Lambda function
    # Squeeze to (batch, 16) so softmax goes across time steps
    score_vec_squeezed = tf.squeeze(x, axis=-1)
    attention_weights = tf.nn.softmax(score_vec_squeezed, axis=1)  # shape: (batch, 16)
    # Expand back to (batch, 16, 1)
    attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
    return attention_weights_expanded

def reduce_sum_custom(x):
    import tensorflow as tf  # Explicitly import TensorFlow inside the Lambda function
    return tf.reduce_sum(x, axis=1)

# Load the test datax
X_test = np.load('./datasets/news_data_X_test.npy')
Y_test = np.load('./datasets/news_data_Y_test.npy')

# Load the saved model from the .h5 file, passing the custom Lambda functions in custom_objects
model = tf.keras.models.load_model('./models/final_model.h5', custom_objects={
    'attention_weighted_sum': attention_weighted_sum,
    'reduce_sum_custom': reduce_sum_custom
})

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)

# Print the results
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
