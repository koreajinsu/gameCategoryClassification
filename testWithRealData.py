import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Text preprocessing for English
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random


# NLTK 자연어 처리 기본 도구 다운로드 받기
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------------------------------
# Custom Lambda Functions (if your model requires them)
# --------------------------------------------------
def attention_weighted_sum(x):
    import tensorflow as tf  # Import inside lambda
    score_vec_squeezed = tf.squeeze(x, axis=-1)  # shape: (batch, time_steps)
    attention_weights = tf.nn.softmax(score_vec_squeezed, axis=1)  # shape: (batch, time_steps)
    attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)  # shape: (batch, time_steps, 1)
    return attention_weights_expanded

def reduce_sum_custom(x):
    import tensorflow as tf
    return tf.reduce_sum(x, axis=1)

# --------------------------------------------------
# 1) Load Your Model, Tokenizer, and LabelEncoder
# --------------------------------------------------
model_path = './models/final_model.h5'
tokenizer_path = './models/tokenizer.pickle'
encoder_path   = './models/encoder.pickle'


# Load the saved model from the .h5 file, passing the custom Lambda functions in custom_objects
model = tf.keras.models.load_model('./models/final_model.h5', custom_objects={
    'attention_weighted_sum': attention_weighted_sum,
    'reduce_sum_custom': reduce_sum_custom
})


# Load the tokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open(encoder_path, 'rb') as handle:
    label_encoder = pickle.load(handle)

# --------------------------------------------------
# 2) Load New Data (XYZ.csv)
# --------------------------------------------------
df_new = pd.read_csv('./datasets/filtered_reviews_with_word_count_200-300.csv')  # Make sure your CSV has 'review' and 'genre' columns

# Remove duplicates just in case
df_new.drop_duplicates(inplace=True)
df_new.reset_index(drop=True, inplace=True)

# --------------------------------------------------
# 3) (Optionally) Sample 1000 Rows Randomly
# --------------------------------------------------
# If you have more than 1000 rows and want to randomly pick 1000:
if len(df_new) > 1000:
    df_new = df_new.sample(n=1000, random_state=42).reset_index(drop=True)

# Extract the columns
X_real = df_new['review']
Y_real = df_new['genre']

# --------------------------------------------------
# 4) Define Preprocessing (Same as Training)
# --------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(sentence):
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.isalpha()]      # Keep only alphabetic words
    tokens = [word for word in tokens if word not in stop_words]      # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]          # Lemmatize words
    return tokens

# Preprocess all reviews
X_preprocessed = X_real.apply(lambda x: ' '.join(preprocess_text(x)))

# --------------------------------------------------
# 5) Convert Text to Sequences and Pad
# --------------------------------------------------
tokenized_X = tokenizer.texts_to_sequences(X_preprocessed)

# To match the padding length used during training, we can:
#  - Either load the same 'max_length' you used
#  - Or compute here. If you used `max_length = max(len(seq) for seq in tokenized_X)` earlier,
#    you need to ensure it matches EXACTLY what the model was trained on.

# For demonstration, let's assume we stored the max length in a variable or we just re-use the same approach:
# max_length = max(len(seq) for seq in tokenized_X)  # or you can hardcode if you know your training max length

max_length = 297  # or whatever you used in training
X_pad_new = pad_sequences(tokenized_X, maxlen=max_length, padding='post')

# --------------------------------------------------
# 6) Predict on New Data
# --------------------------------------------------
pred_probabilities = model.predict(X_pad_new)  # shape: (num_samples, num_classes)
pred_indices = np.argmax(pred_probabilities, axis=1)

# Decode predicted labels back to original category names
pred_labels = label_encoder.inverse_transform(pred_indices)

# --------------------------------------------------
# 7) Compare Predictions with True Labels
# --------------------------------------------------
# Convert real labels to numeric for comparison
true_indices = label_encoder.transform(Y_real)

# Calculate accuracy
accuracy = np.mean(pred_indices == true_indices)
print(f"Test Accuracy on XYZ.csv (sample of {len(df_new)}): {accuracy:.4f}")

# Optionally, show some random samples of predictions
print("\n-- Random Samples of Predictions --")
random_samples = random.sample(range(len(df_new)), min(10, len(df_new)))  # 10 random samples
for idx in random_samples:
    print(f"Review: {df_new.loc[idx, 'review']}")
    print(f" True Genre   : {df_new.loc[idx, 'genre']}")
    print(f" Predicted Gen: {pred_labels[idx]}")
    print("--------------")
