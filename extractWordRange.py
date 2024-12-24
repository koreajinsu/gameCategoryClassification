import pandas as pd

# 1. Load the CSV that already contains the 'word_count_range' column
df = pd.read_csv("./datasets/filtered_reviews_with_word_count.csv")

# 2. Remove rows with null values in 'genre'
df = df.dropna(subset=['genre'])

# 3. Get the unique ranges
unique_ranges = df["word_count_range"].unique()

# 4. For each range, create a new CSV containing only rows with that range
for rng in unique_ranges:
    # Optional: sanitize the range string for safe filenames
    # e.g., replace '+' with 'plus'
    safe_rng = rng.replace("+", "plus")

    # Filter rows
    df_range = df[df["word_count_range"] == rng]

    # 5. Save to CSV, for example: filtered_reviews_with_word_count_0-100.csv
    out_filename = f"./datasets/filtered_reviews_with_word_count_{safe_rng}.csv"
    df_range.to_csv(out_filename, index=False)
    print(f"Saved {out_filename} with {len(df_range)} rows.")
