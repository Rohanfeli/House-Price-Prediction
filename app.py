from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd  # Import pandas to handle DataFrame conversion
import pickle

app = Flask(__name__)

# Load the saved model pipeline
with open('model_pipeline.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

# Serve the HTML form for input
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Extract input values and prepare them in a dictionary
    input_data = {
        'longitude': float(data['longitude']),
        'latitude': float(data['latitude']),
        'housing_median_age': float(data['housing_median_age']),
        'total_rooms': float(data['total_rooms']),
        'total_bedrooms': float(data['total_bedrooms']),
        'population': float(data['population']),
        'households': float(data['households']),
        'median_income': float(data['median_income']),
        'ocean_proximity': data['ocean_proximity']
    }

    # Convert input data to DataFrame to satisfy the pipeline's requirement for column names
    input_df = pd.DataFrame([input_data])

    # Make a prediction using the loaded model
    predicted_price = model_pipeline.predict(input_df)[0]

    # Redirect to result page with prediction
    return redirect(url_for('result', price=round(predicted_price, 2)))

# Result page
@app.route('/result')
def result():
    price = request.args.get('price')
    return render_template('result.html', predicted_price=price)

if __name__ == '__main__':
    app.run(debug=True)
