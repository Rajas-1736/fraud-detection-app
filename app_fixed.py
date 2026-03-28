from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("🔥 Predict route hit")

    try:
        # Get input values
        time = float(request.form.get('Time', 0))
        amount = float(request.form.get('Amount', 0))

        print(f"Time: {time}, Amount: {amount}")

        # Create 30-feature input (standard CC fraud: Time + V1-V28 + Amount)
        features = np.zeros((1, 30))
        features[0, 0] = time  # Time
        features[0, -1] = amount  # Amount at index 29

        print(f"Feature shape: {features.shape}")

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        print(f"Prediction: {prediction}, Proba: {probability}")

        if prediction == 1:
            result = f"Fraud Transaction 🚨 (Probability: {probability[1]:.2%})"
            result_class = "fraud"
        else:
            result = f"Normal Transaction ✅ (Probability: {probability[1]:.2%})"
            result_class = "normal"

        return render_template('index.html', prediction_text=result, result_class=result_class)

    except Exception as e:
        print(f"ERROR: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}", result_class="error")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

