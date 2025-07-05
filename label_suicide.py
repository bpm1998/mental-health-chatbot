import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load cleaned data
df = pd.read_csv("reddit_depression_cleaned.csv")
texts = df["clean_selftext"].fillna("").tolist()

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels
candidate_labels = ["suicidal", "not suicidal"]

# Create column for predictions
suicide_labels = []

print("🔍 Labeling posts for suicidal intent...")

# Loop through posts with progress bar
for text in tqdm(texts):
    result = classifier(text, candidate_labels)
    suicide_labels.append(result["labels"][0])  # Top prediction

# Add column
df["suicidal_intent"] = suicide_labels

# Save
df.to_csv("reddit_labeled_suicide.csv", index=False, encoding="utf-8")
print("✅ Done! Saved labeled data to reddit_labeled_suicide.csv")
