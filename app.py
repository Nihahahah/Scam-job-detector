from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load model and vectorizer
model = joblib.load("scam_rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("message", "")

    if not text.strip():
        return jsonify({"label": "❌ Empty input!", "confidence": 0.0})

    x_input = vectorizer.transform([text])
    pred = model.predict(x_input)[0]
    conf = model.predict_proba(x_input)[0][pred] * 100

    # Enhanced labeling logic based on prediction and confidence
    if pred == 1:
        label = "🚨 Scam Job"
    elif conf < 60:
        label = "⚠️ Possibly Scam - Low Confidence"
    else:
        label = "✅ Legit Job"

    return jsonify({
        "label": label,
        "confidence": round(conf, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
