import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# App UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it's **spam or not**")

# Input
user_input = st.text_area("✉️ Enter your message here:", height=150)

if st.button("🔍 Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Preprocess and predict
        cleaned = preprocess(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM!**")
            st.write(f"Confidence: **{round(probability[1] * 100, 2)}%**")
        else:
            st.success("✅ This message is **NOT SPAM!**")
            st.write(f"Confidence: **{round(probability[0] * 100, 2)}%**")