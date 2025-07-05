import joblib
import re
import pandas as pd
from datetime import datetime
import os
from deep_translator import GoogleTranslator
from langdetect import detect

# Load model and vectorizer
model = joblib.load("suicide_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
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

def clean_text(text):
    # Keep English and Hindi
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Start bot
print("🤖 Mental Health Bot: Hello! I'm here to listen. (type 'exit' to quit)\n")

while True:
    user_input = input("🧑 You: ")
    if user_input.lower() == "exit":
        print("🤖 Bot: Take care! You're not alone. 💙")
        break

    # 1. Detect language
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = 'en'

    # 2. Translate to English if not English
    if detected_lang != 'en':
        try:
            english_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        except:
            english_input = user_input
    else:
        english_input = user_input

    # 3. Predict
    clean_input = clean_text(english_input)
    vectorized_input = vectorizer.transform([clean_input])
    prediction = model.predict(vectorized_input)[0]

    # 4. Generate English response
    if prediction == 1:
        response_en = "I'm really sorry you're feeling this way. You're not alone. 💙 Please consider talking to someone or call 9152987821 (helpline)."
    else:
        response_en = "That’s great to hear! 😊 I'm always here to chat."

    # 5. Translate response if needed
    if detected_lang == 'hi':
        try:
            final_response = GoogleTranslator(source='en', target='hi').translate(response_en)
        except:
            final_response = response_en
    else:
        final_response = response_en

    # 6. Output & log
    print(f"\n🤖 Bot: {final_response}")
    log_interaction(user_input, prediction, final_response)
