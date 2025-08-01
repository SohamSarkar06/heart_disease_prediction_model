from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Heart Disease Predictor API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract features in correct order
    features = np.array([[
        data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
        data["fbs"], data["restecg"], data["thalach"], data["exang"],
        data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]])

    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # for Render
