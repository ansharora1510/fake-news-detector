import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Function to clean the text (same as training)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

# Streamlit app interface
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter the news article or headline:")

if st.button("Check"):
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)

    if prediction[0] == 0:
        st.error("🚨 This news is *FAKE*.")
    else:
        st.success("✅ This news is *REAL*.")
