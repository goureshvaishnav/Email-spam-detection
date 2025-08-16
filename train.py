
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Standardize label to 0/1
    df['label'] = df['label'].str.strip().str.lower().map({'ham':0, 'spam':1})
    df = df.dropna(subset=['text', 'label'])
    return df

def build_pipeline():
    # Simple, fast baseline for text classification
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=1)),
        ("nb", MultinomialNB())
    ])
    return pipe

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    model_path = os.path.join(MODELS_DIR, "spam_model.joblib")
    joblib.dump(pipe, model_path)

    # Save metrics
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to:", model_path)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()
