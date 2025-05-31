import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
import os
warnings.filterwarnings("ignore")

# Load the dataset
try:
    # Try to load from the data directory first
    data_path = os.path.join('data', 'nutritional_requirements_extended.csv')
    
    # If not found in data directory, try the nutrtion app directory
    if not os.path.exists(data_path):
        data_path = os.path.join('nutrtion app', 'nutritional_requirements_extended.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Could not find the dataset file. Please ensure it's in the 'data' or 'nutrtion app' directory.")
    
    data = pd.read_csv(data_path)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise

# Strip column names to avoid trailing spaces
data.rename(columns=lambda x: x.strip(), inplace=True)

# Label Encoding for 'Sex' and 'Activity Level'
label_encoder_sex = LabelEncoder()
label_encoder_activity = LabelEncoder()

data['Sex'] = label_encoder_sex.fit_transform(data['Sex'])
data['Activity Level'] = label_encoder_activity.fit_transform(data['Activity Level'])

# Replace 'NULL' and NaN in 'Pregnant' with 0 and convert to integer
data['Pregnant'] = data['Pregnant'].replace(['NULL', np.nan], 0).astype(int)

# Features and Targets
features = ['Age', 'Height (cm)', 'Weight (kg)', 'Sex', 'Activity Level', 'Pregnant']
targets = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fats (g)', 'Water (L)',
           'Vitamin A (mcg)', 'Vitamin B12 (mcg)', 'Vitamin C (mg)', 'Vitamin D (IU)',
           'Calcium (mg)', 'Iron (mg)', 'Magnesium (mg)', 'Zinc (mg)', 'Omega-3 (mg)']

X = data[features]
y = data[targets]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor for each target
models = {}
for target in targets:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[target])
    models[target] = model

# Function to get user input and predict nutritional requirements
def predict_nutritional_requirements(age, height, weight, activity, sex, pregnant=0):
    try:
        # Encode the inputs
        sex_encoded = label_encoder_sex.transform([sex])[0]
        activity_encoded = label_encoder_activity.transform([activity])[0]
        
        # Create input array
        user_input = np.array([[age, height, weight, sex_encoded, activity_encoded, pregnant]])
        
        # Predict all targets
        predictions = {}
        for target, model in models.items():
            predictions[target] = float(model.predict(user_input)[0])
        
        return predictions
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

if __name__ == '__main__':
    # Example usage when running the script directly
    print("Please enter the following details:")
    try:
        age = int(input("Age: "))
        height = float(input("Height (cm): "))
        weight = float(input("Weight (kg): "))
        activity = input("Physical Activity (Sedentary, Moderate, Heavy): ").capitalize()
        sex = input("Sex (Male/Female): ").capitalize()
        
        pregnant = 0
        if sex == 'Female':
            pregnant_input = input("Are you pregnant? (Yes/No): ").capitalize()
            pregnant = 1 if pregnant_input == 'Yes' else 0
            
        predictions = predict_nutritional_requirements(age, height, weight, activity, sex, pregnant)
        
        if predictions:
            print("\nNutritional Requirements:")
            for key, value in predictions.items():
                print(f"{key}: {value:.2f}")
    except ValueError:
        print("Invalid input. Please enter numeric values for age, height, and weight.")