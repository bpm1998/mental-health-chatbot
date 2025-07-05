import streamlit as st
from langdetect import detect
from deep_translator import GoogleTranslator
import joblib
import re
from transformers import pipeline

# Load Hugging Face emotion classifier
#emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

@st.cache_resource
def load_emotion_model():
    from transformers import pipeline
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

emotion_classifier = load_emotion_model()

# Load model and vectorizer
model = joblib.load("suicide_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_emotion(text):
    try:
        result = emotion_classifier(text)[0][0]
        return result['label'], result['score']
    except:
        return "Unknown", 0.0

def generate_response(user_input):
    try:
        lang = detect(user_input)
    except:
        lang = 'en'

    if lang != 'en':
        english_input = GoogleTranslator(source='auto', target='en').translate(user_input)
    else:
        english_input = user_input

    # Detect emotion
    emotion, score = detect_emotion(english_input)

    clean_input = clean_text(english_input)
    vectorized_input = vectorizer.transform([clean_input])
    prediction = model.predict(vectorized_input)[0]

    if prediction == 1:
        response_en = "I'm really sorry you're feeling this way. You're not alone. 💙 Please consider talking to someone or call 9152987821 (helpline)."
    else:
        response_en = "That’s great to hear! 😊 I'm always here to chat."

    if lang == 'hi':
        translated_response = GoogleTranslator(source='en', target='hi').translate(response_en)
        return translated_response, emotion
    else:
        return response_en, emotion

# UI
st.title("🤖 Mental Health Chatbot")
st.write("Enter your message below (English or Hindi):")

user_input = st.text_input("🧑 You:")

if st.button("Send") and user_input:
    bot_reply, emotion = generate_response(user_input)
    st.markdown(f"**🤖 Bot:** {bot_reply}")
    st.markdown(f"🧠 **Detected Emotion:** `{emotion}`")
