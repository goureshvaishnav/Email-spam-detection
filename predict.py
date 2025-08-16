
import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run: python src/train.py")
        sys.exit(1)
    return joblib.load(MODEL_PATH)

def predict_text(text: str):
    model = load_model()
    pred = model.predict([text])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0].tolist()
    label = "spam" if pred == 1 else "ham"
    return label, proba

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "Your email text here"')
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    label, proba = predict_text(text)
    print(f"Prediction: {label}")
    if proba is not None:
        print(f"Probabilities [ham, spam]: {proba}")
