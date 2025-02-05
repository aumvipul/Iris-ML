import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load model, scaler, and label encoder
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        print("Received input data:", data)  # Add this line to check the input data
        data = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data)
        prediction_index = model.predict(data_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_index])[0]  # Convert number back to species

        return jsonify({"prediction": prediction_label})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
