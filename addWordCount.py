import pandas as pd
import numpy as np

df_filtered_game_reviews = pd.read_csv('./datasets/filtered_reviews_with_genre.csv')

print(df_filtered_game_reviews.head())
print(df_filtered_game_reviews.info())
# print(df_game_reviews.category.value_counts())

# 1. Handle missing reviews by filling NaNs with an empty string (optional)
df_filtered_game_reviews['review'] = df_filtered_game_reviews['review'].fillna('')

# 2. Compute the word count for each review
df_filtered_game_reviews['word_count'] = df_filtered_game_reviews['review'].str.split().str.len()

# 3. Define a function to map word counts to the specified ranges
def categorize_word_count(count):
    if count < 100:
        return '0-100'
    elif count < 200:
        return '100-200'
    elif count < 300:
        return '200-300'
    elif count < 400:
        return '300-400'
    elif count < 500:
        return '400-500'
    else:
        return '500+'


# 1. Load CSV
df_filtered_game_reviews = pd.read_csv("./datasets/filtered_reviews_with_genre.csv")

# 2. Fill missing reviews with empty strings
df_filtered_game_reviews["review"] = df_filtered_game_reviews["review"].fillna("")

# 3. Create a new column: number of words in each review
df_filtered_game_reviews["review_words_count"] = (
    df_filtered_game_reviews["review"].str.split().str.len()
)

# 4. Categorize word counts using the new column
df_filtered_game_reviews["word_count_range"] = df_filtered_game_reviews["review_words_count"].apply(categorize_word_count)

# 5. (Optional) Check distribution
print(df_filtered_game_reviews["word_count_range"].value_counts())

# 6. Save to CSV with the new columns
df_filtered_game_reviews.to_csv("./datasets/filtered_reviews_with_word_count.csv", index=False)