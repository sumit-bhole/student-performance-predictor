from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return " Student Performance Predictor API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        hours = data["hours_studied"]
        attendance = data["attendance_percent"]
        previous = data["previous_score"]
    except (TypeError, KeyError):
        return jsonify({"error": "Invalid input format"}), 400

    input_data = np.array([[hours, attendance, previous]])
    predicted = model.predict(input_data)[0]

    return jsonify({
        "predicted_marks": round(predicted, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
