import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Load and combine both datasets
df_real = pd.read_csv("reddit_labeled_suicide.csv")[["clean_selftext", "suicidal_intent"]]
df_synth = pd.read_csv("synthetic_suicide_data.csv")[["clean_selftext", "suicidal_intent"]]

# Rename synthetic dataset columns to match
df_synth.columns = ["clean_selftext", "suicidal_intent"]

# Combine datasets
df = pd.concat([df_real, df_synth], ignore_index=True)

# Clean labels
df["suicidal_intent"] = df["suicidal_intent"].astype(str).str.strip().str.lower()
df["label"] = df["suicidal_intent"].map({"suicidal": 1, "not suicidal": 0})

print("\n📊 Dataset Loaded:")
print(df["label"].value_counts())
print("Total records:", len(df))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["cleaned_text"] = df["clean_selftext"].astype(str).apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF vectorizer with n-grams
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Suicidal", "Suicidal"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Suicidal", "Suicidal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model and vectorizer
joblib.dump(model, "suicide_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Logistic Regression model and vectorizer saved successfully.")
