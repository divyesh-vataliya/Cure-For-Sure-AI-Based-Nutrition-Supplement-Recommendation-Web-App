from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load your pre-trained models (should be trained and saved beforehand)
def load_models():
    # This assumes you've saved your trained models using joblib
    models = {}
    targets = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fats (g)', 'Water (L)',
               'Vitamin A (mcg)', 'Vitamin B12 (mcg)', 'Vitamin C (mg)', 'Vitamin D (IU)',
               'Calcium (mg)', 'Iron (mg)', 'Magnesium (mg)', 'Zinc (mg)', 'Omega-3 (mg)']
    
    for target in targets:
        models[target] = joblib.load(f'models/{target}_model.joblib')
    
    return models

models = load_models()
label_encoder_sex = LabelEncoder()
label_encoder_sex.fit(['Male', 'Female'])
label_encoder_activity = LabelEncoder()
label_encoder_activity.fit(['Sedentary', 'Moderate', 'Heavy'])

# Supplement database (simplified example)
supplement_db = {
    'Vitamin D': {
        'description': 'Essential for bone health and immune function',
        'food_sources': ['Fatty fish (salmon, tuna)', 'Egg yolks', 'Fortified dairy products'],
        'recommended': 'Consider supplements if sun exposure is limited'
    },
    # Add more supplements as needed
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Add actual authentication logic here
        session['logged_in'] = True
        return redirect(url_for('calculator'))
    return render_template('login.html')

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Process form data
        try:
            age = int(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            sex = request.form['sex']
            activity = request.form['activity']
            pregnant = 0
            
            if sex == 'Female':
                pregnant = 1 if request.form.get('pregnant') == 'on' else 0
            
            # Encode inputs
            sex_encoded = label_encoder_sex.transform([sex])[0]
            activity_encoded = label_encoder_activity.transform([activity])[0]
            
            # Create input array
            user_input = np.array([[age, height, weight, sex_encoded, activity_encoded, pregnant]])
            
            # Predict all targets
            predictions = {}
            for target, model in models.items():
                predictions[target] = model.predict(user_input)[0]
            
            session['predictions'] = predictions
            return redirect(url_for('results'))
        
        except ValueError:
            error = "Please enter valid numeric values"
            return render_template('calculator.html', error=error)
    
    return render_template('calculator.html')

@app.route('/results')
def results():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    predictions = session.get('predictions', {})
    
    # Determine which supplements might be needed
    supplements_needed = []
    if predictions.get('Vitamin D (IU)', 0) < 600:
        supplements_needed.append('Vitamin D')
    # Add more conditions for other supplements
    
    supplement_info = {}
    for supplement in supplements_needed:
        supplement_info[supplement] = supplement_db.get(supplement, {})
    
    return render_template('results.html', 
                         predictions=predictions, 
                         supplements=supplement_info)

if __name__ == '__main__':
    app.run(debug=True)