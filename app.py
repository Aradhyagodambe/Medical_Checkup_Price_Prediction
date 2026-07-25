import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the Scikit-Learn model
with open('Checkup_Price.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    # Render the input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the HTML form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        smoker = float(request.form['smoker'])
        region = float(request.form['region'])
        
        # Format for the model (2D array)
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Generate prediction
        prediction = model.predict(features)[0]
        
        return render_template('index.html', prediction_text=f'Predicted Charges: ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # Gunicorn will override this in production, but it's useful for local testing
    app.run(host="0.0.0.0", port=8000)
