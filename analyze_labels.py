import pandas as pd

# Load your labeled CSV file
df = pd.read_csv("reddit_labeled_suicide.csv")

# Show label distribution
print("🔍 Suicidal intent label counts:")
print(df["suicidal_intent"].value_counts())
