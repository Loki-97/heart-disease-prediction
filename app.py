from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the final trained model (with all 13 features)
model = joblib.load("heart_disease_model.pkl")

# Define the 13 input feature names (as in training)
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all 13 feature values from the form
        input_values = [float(request.form[feature]) for feature in feature_names]

        # Convert inputs to NumPy array and reshape for prediction
        input_data = np.array(input_values).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert numerical prediction to human-readable output
        result = "Heart Disease Detected 😞" if prediction == 1 else "No Heart Disease 😊"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
