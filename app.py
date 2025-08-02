from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Parse inputs safely
        hours = float(data.get("hours_studied", None))
        attendance = float(data.get("attendance_percent", None))
        previous = float(data.get("previous_score", None))

        # Check for missing or invalid inputs
        if None in [hours, attendance, previous] or any(np.isnan([hours, attendance, previous])):
            return jsonify({"error": "All fields are required and must be numbers."}), 400

    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # Predict
    input_data = np.array([[hours, attendance, previous]])
    predicted = model.predict(input_data)[0]

    return jsonify({"predicted_marks": round(predicted, 2)})

if __name__ == "__main__":
    app.run(debug=True)
