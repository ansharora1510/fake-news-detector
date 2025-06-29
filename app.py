import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
print("Restart trigger")

# Download stopwords if needed
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit App UI
st.title("📰 Fake News Detector")
st.markdown("Enter news text below to check if it's **Fake** or **Real**.")

input_text = st.text_area("Paste News Article Text Here")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        if prediction == 0:
            st.error("🚨 This news article is likely **FAKE**.")
        else:
            st.success("✅ This news article appears to be **REAL**.")
