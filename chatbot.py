import joblib
import re
import pandas as pd
from datetime import datetime
import os
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import pipeline

# Load the emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Load model and vectorizer
try:
    model = joblib.load("suicide_classifier_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    exit()

log_file = "chat_logs.csv"

def log_interaction(user_input, prediction, bot_reply):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = "Suicidal" if prediction == 1 else "Not Suicidal"
    log_entry = {
        "timestamp": timestamp,
        "user_input": user_input,
        "prediction": label,
        "bot_reply": bot_reply
    }

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_file, index=False)


def contains_suicidal_keywords(text):
    keywords = ["kill myself", "end my life", "suicide", "die", "kill me", "murder me", "want to die"]
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)  # keep Hindi + English words
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Start bot
print("🤖 Mental Health Bot: Hello! I'm here to listen. (type 'exit' to quit)\n")

while True:
    user_input = input("🧑 You: ").strip()
    if user_input.lower() == "exit":
        print("🤖 Bot: Take care! You're not alone. 💙")
        break

    # 1. Detect language
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = 'en'

    # 2. Translate to English if not already
    if detected_lang != 'en':
        try:
            english_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        except:
            english_input = user_input
    else:
        english_input = user_input

    # 3. Detect emotion
    try:
        emotion_result = emotion_classifier(english_input)[0][0]
        detected_emotion = emotion_result['label'].lower()
       # print(f"[DEBUG] Detected emotion: {detected_emotion}, Language: {detected_lang}, Suicide prob: {suicide_prob:.2f}")

    except Exception as e:
        detected_emotion = "unknown"

    # 4. Suicide intent prediction
    clean_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([clean_input])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vectorized_input)[0]
        suicide_prob = proba[1]
    else:
        suicide_prob = model.predict(vectorized_input)[0]
    print(f"[DEBUG] Detected emotion: {detected_emotion}, Language: {detected_lang}, Suicide prob: {suicide_prob}")

    # 4. Final prediction based on both model & keywords
    threshold = 0.6
    prediction = 1 if suicide_prob >= threshold or contains_suicidal_keywords(user_input) else 0

    # 5. Response generation
    if prediction == 1:
        response = "I'm really sorry you're feeling this way. You're not alone. 💙 Please consider talking to someone or call 9152987821 (helpline)."
    elif detected_emotion in ["sadness", "anger", "fear", "disgust"]:
        response = "I'm sorry you're feeling down. I'm here to support you. 💙 You're not alone."
    elif any(word in english_input.lower() for word in ["sad", "depressed", "hopeless", "low", "meaningless", "lonely", "worthless"]):
        response = "It sounds like you're going through something. 💙 Please know you're not alone."
    else:
        response = "That’s great to hear! 😊 I'm always here to chat."

    # 6. Translate response back to Hindi if needed
    if detected_lang == 'hi':
        try:
            final_response = GoogleTranslator(source='en', target='hi').translate(response)
        except:
            final_response = response
    else:
        final_response = response

    # 7. Output & log
    print(f"\n🤖 Bot: {final_response}")
    log_interaction(user_input, prediction, final_response)
