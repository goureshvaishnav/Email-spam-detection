
# Email Spam Detection (Fast Internship Project)

A simple, presentation-ready project: **TF-IDF + Naive Bayes** to detect spam vs ham.

## Project Structure
```
spam_detection_project/
├─ data/
│  └─ spam.csv               # sample dataset (replace with a larger one later)
├─ models/                   # trained model & metrics will be saved here
├─ src/
│  ├─ train.py               # train & evaluate model
│  └─ predict.py             # CLI prediction
├─ app.py                    # Streamlit demo app
└─ requirements.txt
```

## Setup (Windows)
1. Install Python 3.10+ from https://python.org (if not already installed).
2. Open **Command Prompt** and create a virtual environment:
   ```bat
   py -m venv venv
   venv\Scripts\activate
   ```
3. Go to the project folder and install dependencies:
   ```bat
   pip install -r requirements.txt
   ```

## Train the Model
```bat
python src\train.py
```
This will save `models\spam_model.joblib` and `models\metrics.json`.

## Quick Test (CLI)
```bat
python src\predict.py "Win a free iPhone! Click to claim now"
```

## Run the App (UI Demo)
```bat
streamlit run app.py
```
Then open the URL shown (usually http://localhost:8501).

## Replace with a Bigger Dataset (Optional but Recommended)
- Replace `data\spam.csv` with a larger dataset (two columns: `label` in {ham, spam} and `text`).
- Keep the column names the same.

## Notes
- Baseline model: **TfidfVectorizer(ngram 1–2, english stopwords) + MultinomialNB**
- Great for internship presentation: simple pipeline, high-level accuracy, easy demo.
