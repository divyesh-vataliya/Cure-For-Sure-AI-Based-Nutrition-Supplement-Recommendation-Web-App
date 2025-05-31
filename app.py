from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import json

# Import the model from n2.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from n2 import models, label_encoder_sex, label_encoder_activity
    print("Successfully imported model from n2.py")
except Exception as e:
    print(f"Error importing model from n2.py: {str(e)}")
    models = None
    label_encoder_sex = None
    label_encoder_activity = None

app = Flask(__name__)
app.secret_key = 'cure_for_sure_secret_key'  # Change this to a secure secret key in production

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple user class for demonstration
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# In-memory user storage (replace with database in production)
users = {}
user_id_counter = 1

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Find user
        user = next((u for u in users.values() if u.username == username), None)
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if username already exists
        if any(u.username == username for u in users.values()):
            flash('Username already exists')
            return redirect(url_for('register'))
        
        # Create new user
        global user_id_counter
        user = User(user_id_counter, username, generate_password_hash(password))
        users[user_id_counter] = user
        user_id_counter += 1
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if models is None:
        flash('Error: The nutrition prediction model is not available. Please contact support.')
        return render_template('dashboard.html')
        
    if request.method == 'POST':
        try:
            age = int(request.form.get('age'))
            height = float(request.form.get('height'))
            weight = float(request.form.get('weight'))
            activity = request.form.get('activity')
            sex = request.form.get('sex')
            pregnant = 1 if request.form.get('pregnant') == 'yes' and sex == 'Female' else 0
            
            # Create input array
            user_input = np.array([[age, height, weight, 
                                  label_encoder_sex.transform([sex])[0],
                                  label_encoder_activity.transform([activity])[0], 
                                  pregnant]])
            
            # Predict all targets
            predictions = {}
            for target, model in models.items():
                predictions[target] = float(model.predict(user_input)[0])
            
            return render_template('dashboard.html', predictions=predictions)
        except Exception as e:
            flash(f'Error making prediction: {str(e)}')
            return render_template('dashboard.html')
    
    return render_template('dashboard.html')

@app.route('/supplements')
@login_required
def supplements():
    supplement_data = {
        'Vitamin A': {
            'natural_sources': ['Carrots', 'Sweet potatoes', 'Spinach', 'Kale', 'Liver'],
            'benefits': ['Vision health', 'Immune system', 'Skin health']
        },
        'Vitamin B12': {
            'natural_sources': ['Fish', 'Meat', 'Eggs', 'Dairy products'],
            'benefits': ['Red blood cell formation', 'Nerve function', 'DNA synthesis']
        },
        'Vitamin C': {
            'natural_sources': ['Oranges', 'Strawberries', 'Bell peppers', 'Broccoli'],
            'benefits': ['Immune system', 'Collagen formation', 'Antioxidant']
        },
        'Vitamin D': {
            'natural_sources': ['Sunlight', 'Fatty fish', 'Egg yolks', 'Fortified dairy'],
            'benefits': ['Bone health', 'Calcium absorption', 'Immune function']
        },
        'Calcium': {
            'natural_sources': ['Dairy products', 'Leafy greens', 'Sardines', 'Tofu'],
            'benefits': ['Bone health', 'Muscle function', 'Nerve signaling']
        },
        'Iron': {
            'natural_sources': ['Red meat', 'Beans', 'Spinach', 'Fortified cereals'],
            'benefits': ['Red blood cell formation', 'Oxygen transport', 'Energy production']
        },
        'Magnesium': {
            'natural_sources': ['Nuts', 'Seeds', 'Legumes', 'Whole grains', 'Dark chocolate'],
            'benefits': ['Muscle and nerve function', 'Energy production', 'Bone health']
        },
        'Zinc': {
            'natural_sources': ['Oysters', 'Meat', 'Legumes', 'Seeds', 'Nuts'],
            'benefits': ['Immune function', 'Wound healing', 'DNA synthesis']
        },
        'Omega-3': {
            'natural_sources': ['Fatty fish', 'Flaxseeds', 'Chia seeds', 'Walnuts'],
            'benefits': ['Heart health', 'Brain function', 'Reduced inflammation']
        }
    }
    return render_template('supplements.html', supplement_data=supplement_data)

if __name__ == '__main__':
    app.run(debug=True) 