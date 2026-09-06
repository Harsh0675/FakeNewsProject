from flask import Flask, render_template, request, jsonify
from fake_news import load_model, predict_text

app = Flask(__name__)

try:
    pipeline = load_model()
except Exception:
    pipeline = None

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})

@app.post("/api/predict")
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not trained. Run: python fake_news.py --train"}), 503
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Enter a news claim or article."}), 400
    result = predict_text(pipeline, text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
