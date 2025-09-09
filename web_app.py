from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained SVM model for Medical Based Prediction
svm_model = joblib.load('models/svm_heart_disease_model.pkl')




# ✅ Route for Landing Page
@app.route('/')
def index():
    return render_template('landing_page.html')


# ✅ Route for Medical Report-Based Prediction 
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [float(request.form[key]) for key in request.form.keys()]
        input_data = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = svm_model.predict(input_data)
        result = "Heart Disease Detected !" if prediction[0] == 1 else "No Heart Disease Detected🤗."

        # Return JSON response
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})      
        
    
if __name__ == '__main__':
    app.run(debug=True)