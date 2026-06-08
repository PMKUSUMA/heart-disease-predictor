from flask import Blueprint, render_template, request, jsonify
import pickle
import numpy as np
import csv
from app.models import db, PredictionLog
import os

main: Blueprint = Blueprint("main", __name__)

MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "model", "tuned_best_classifier.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    try:
        features: list[float] = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"]),
        ]

        input_array: np.ndarray = np.array(features).reshape(1, -1)
        prediction: int = int(model.predict(input_array)[0])
        probability: float = round(float(model.predict_proba(input_array)[0][prediction]) * 100, 2)

        log: PredictionLog = PredictionLog(
            age=features[0],
            sex=features[1],
            prediction=prediction,
            confidence=probability
        )
        db.session.add(log)
        db.session.commit()

        return render_template("result.html", prediction=prediction, probability=probability)

    except Exception as e:
        return render_template("result.html", error=str(e))


@main.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data: dict = request.get_json()

        features: list[float] = [
            float(data["age"]),
            float(data["sex"]),
            float(data["cp"]),
            float(data["trestbps"]),
            float(data["chol"]),
            float(data["fbs"]),
            float(data["restecg"]),
            float(data["thalach"]),
            float(data["exang"]),
            float(data["oldpeak"]),
            float(data["slope"]),
            float(data["ca"]),
            float(data["thal"]),
        ]

        input_array: np.ndarray = np.array(features).reshape(1, -1)
        prediction: int = int(model.predict(input_array)[0])
        probability: float = round(float(model.predict_proba(input_array)[0][prediction]) * 100, 2)

        log: PredictionLog = PredictionLog(
            age=features[0],
            sex=features[1],
            prediction=prediction,
            confidence=probability
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            "prediction": prediction,
            "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "confidence": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@main.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": "Heart Disease Predictor"})


@main.route("/api/history", methods=["GET"])
def history():
    logs: list[PredictionLog] = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(20).all()
    return jsonify([{
        "id": log.id,
        "age": log.age,
        "sex": log.sex,
        "prediction": log.prediction,
        "confidence": log.confidence,
        "timestamp": str(log.timestamp)
    } for log in logs])


DATA_PATH: str = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")


@main.route("/data")
def dataset_view():
    rows: list[dict] = []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return render_template("data.html", rows=rows, columns=list(rows[0].keys()) if rows else [])


@main.route("/api/data")
def api_dataset():
    rows: list[dict] = []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify({"count": len(rows), "columns": list(rows[0].keys()) if rows else [], "data": rows})