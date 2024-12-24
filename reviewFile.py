import pandas as pd
import numpy as np

# df_games_description = pd.read_csv('./datasets/games_description.csv')



#카테고리 수정된 파일 불러오기
filtered_games_with_alt = pd.read_csv('./datasets/filtered_games_with_alt.csv')
#컬럼 2개 남기고 삭제된 파일 불러오기
df_filtered_game_reviews = pd.read_csv('./datasets/filtered_reviews.csv')

# print(df_games_description.head())
# print(df_games_description.info())
# print(df_games_description.category.value_counts())


print(df_filtered_game_reviews.head())
print(df_filtered_game_reviews.info())
# print(df_game_reviews.category.value_counts())

print(filtered_games_with_alt.head())
print(filtered_games_with_alt.info())
# print(df_game_reviews.category.value_counts())

print("df_filtered_game_reviews columns:", df_filtered_game_reviews.columns)
print("filtered_games_with_alt columns:", filtered_games_with_alt.columns)

# exit()

# 2) Merge the two DataFrames on the common key
#    Assume the common column is named 'name' (adjust if yours is different).
# 2) Merge: left_on='game_name', right_on='name'
merged_df = df_filtered_game_reviews.merge(
    filtered_games_with_alt[['name', 'genres_ALT']],
    left_on='game_name',   # column in df_filtered_game_reviews
    right_on='name',       # column in filtered_games_with_alt
    how='left'
)

# 3) Rename "genres_ALT" to "genre" (if you want that exact name)
merged_df.rename(columns={'genres_ALT': 'genre'}, inplace=True)

# 4) (Optional) If you no longer need the 'name' column after merging, you can drop it:
merged_df.drop(columns=['name'], inplace=True)

# 5) Save the new CSV
merged_df.to_csv('./datasets/filtered_reviews_with_genre.csv', index=False)

print("New file saved: './datasets/filtered_reviews_with_genre.csv'")
