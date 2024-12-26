import pandas as pd

# 파일 경로 로드
file_path = './datasets/filtered_games.csv'

# 데이터셋 불러오기
df = pd.read_csv(file_path)

# 정의된 카테고리 리스트
categories = ['Action', 'RPG', 'Strategy', 'Adventure', 'Simulation', 'FPS', 'Horror', 'Survival', 'Sports', 'Racing',
               'Shooter']


# 새로운 열 생성: genres_ALT
def find_first_category(genres_str: str):
    # Handle NaN or empty
    if pd.isna(genres_str) or not genres_str.strip():
        return None

    # Split by comma into chunks, and strip whitespace
    elements = [s.strip() for s in genres_str.split(',')]

    # Now go through each chunk in order
    for element in elements:
        # Check it against our categories
        for category in categories:
            # If the category is found in the chunk (case-insensitive), return it
            if category.lower() in element.lower():
                return category
    # If no match was found in any chunk, return None
    return None


# Apply to your DataFrame
df['genres_ALT'] = df['genres'].apply(find_first_category)

#새로 추가된 데이터 확인
print(df.head())
df.info()

# CSV 파일 저장
output_path = './datasets/filtered_games_with_alt.csv'
df.to_csv(output_path, index=False)
print(f"저장 완료: {output_path}")