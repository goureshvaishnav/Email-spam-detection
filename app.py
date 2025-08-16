
import os
import json
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.joblib")

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

st.title("📧 Email Spam Detection")
st.write("Type or paste an email/message below and check if it's **Spam** or **Not spam**.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run training first: `python src/train.py`")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

user_text = st.text_area("Enter email/message text:", height=200)

if st.button("Predict", disabled=(model is None)):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        pred = model.predict([user_text])[0]
        label = "Spam" if pred == 1 else "Not spam"
        proba = model.predict_proba([user_text])[0]
        st.subheader(f"Result: **{label}**")
        st.write("Probabilities:")
        st.write({"Not spam": float(proba[0]), "Spam": float(proba[1])})
        st.caption("Helpful to protect against spam emails and phishing attacks (DESIGNED BY Gouresh Vaishnav).")
