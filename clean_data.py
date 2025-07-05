import pandas as pd
import re

# Load the data
df = pd.read_csv("reddit_depression.csv")

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # remove text in brackets (like [removed])
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

# Clean title and selftext
df['clean_title'] = df['title'].apply(clean_text)
df['clean_selftext'] = df['selftext'].apply(clean_text)

# Save cleaned data
df.to_csv("reddit_depression_cleaned.csv", index=False, encoding='utf-8')

print(f"✅ Cleaned data saved to reddit_depression_cleaned.csv with {len(df)} rows.")
