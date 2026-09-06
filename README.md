# Fake News Detector

An educational NLP project that classifies news-style text as **LIKELY REAL** or **LIKELY FAKE** using TF-IDF features and Logistic Regression.

> **Important:** This is a machine-learning demonstration, not a fact-checking authority. A model prediction does not establish whether a claim is true.

## Features

- TF-IDF text feature extraction
- Logistic Regression classifier
- Train/test split and evaluation metrics
- Confidence-style probability output
- Saved model pipeline with Joblib
- Interactive command-line prediction
- Train and predict CLI modes
- Small labeled dataset included for demonstration

## Project structure

```text
FakeNewsProject/
├── data/
│   └── news.csv
├── models/
├── tests/
│   ├── test_fake_news.py
│   └── __init__.py
├── fake_news.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

```bash
python -m pip install -r requirements.txt
python fake_news.py --train
python fake_news.py --predict
```

Or simply:

```bash
python fake_news.py
```

## Example

```text
News > Government announces a new railway project

Prediction : LIKELY REAL
Confidence : 82.4%
```

The confidence value represents the model's estimated class probability, not the factual certainty of the article.

## Testing

```bash
python -m unittest discover -s tests
```

## Limitations

The included dataset is intentionally small and synthetic. Real-world misinformation detection requires substantially larger, diverse, high-quality datasets and should combine model outputs with independent evidence and source verification.
