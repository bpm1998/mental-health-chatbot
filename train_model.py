import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the labeled dataset
df = pd.read_csv("reddit_labeled_suicide.csv")

# Use only clean_selftext and suicidal_intent
df = df[["clean_selftext", "suicidal_intent"]].dropna()

# Rename for convenience
X = df["clean_selftext"]
y = df["suicidal_intent"].map({"suicidal": 1, "not suicidal": 0})  # Convert to binary

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to features
#vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),  # Use unigrams and bigrams
    max_df=0.9,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

#model = LogisticRegression()
model.fit(X_train_vec, y_train)
# print("📊 Classification Report:")
# print(classification_report(y_test, y_pred, target_names=["Not Suicidal", "Suicidal"]))

# Evaluate
y_pred = model.predict(X_test_vec)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Suicidal", "Suicidal"]))

# Save model and vectorizer
joblib.dump(model, "suicide_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
