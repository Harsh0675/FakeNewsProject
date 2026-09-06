import argparse
import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.joblib")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    df = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    return df


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset: {len(df)} articles")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["FAKE", "REAL"], zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training now...\n")
        return train_model()
    return joblib.load(MODEL_PATH)


def predict(model, text: str):
    cleaned = clean_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    probability_map = dict(zip(classes, probabilities))
    real_probability = float(probability_map.get(1, 0.0))
    fake_probability = float(probability_map.get(0, 0.0))
    label = 1 if real_probability >= fake_probability else 0
    return label, real_probability, fake_probability


def interactive(model):
    print("=" * 58)
    print("                 FAKE NEWS DETECTOR")
    print("=" * 58)
    print("Type a news claim. Enter 'exit' to quit.\n")

    while True:
        try:
            text = input("News > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() == "exit":
            print("Goodbye!")
            break
        if not text:
            continue

        label, real_probability, fake_probability = predict(model, text)
        confidence = max(real_probability, fake_probability)
        result = "LIKELY REAL" if label == 1 else "LIKELY FAKE"

        print(f"\nPrediction : {result}")
        print(f"Confidence : {confidence:.1%}")
        print(f"Real       : {real_probability:.1%}")
        print(f"Fake       : {fake_probability:.1%}")
        print("Note       : Model prediction only; verify claims independently.\n")


def main():
    parser = argparse.ArgumentParser(description="Educational fake-news text classifier")
    parser.add_argument("--train", action="store_true", help="train and save the model")
    parser.add_argument("--predict", action="store_true", help="run interactive predictions")
    args = parser.parse_args()

    model = train_model() if args.train else load_model()
    if args.predict or not args.train:
        interactive(model)


if __name__ == "__main__":
    main()
