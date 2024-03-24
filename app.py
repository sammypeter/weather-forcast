from flask_cors import cross_origin
from flask import Flask, request, render_template
import pandas as pd
import pickle
from main import *

app = Flask(__name__)

# Preprocess the raw input data for rain prediction
def preprocess_input(raw_input_values):
    input_df = pd.DataFrame([raw_input_values])

    input_df['RainToday_0'] = (input_df['RainToday'] == 'No').astype(int)
    input_df['RainToday_1'] = (input_df['RainToday'] == 'Yes').astype(int)
    
    input_df = pd.get_dummies(input_df, columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
 
    input_df = input_df[X_train.columns]

    return input_df

@cross_origin()
@app.route('/')
def index():
    return render_template('index1.html')

@cross_origin()
@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    raw_location_temp = request.form['MaxTemp']
    location, temperature = raw_location_temp[:-2], raw_location_temp[-2:]  # Split into location and temperature
    max_temp = float(temperature)  # Convert temperature to float

    raw_input_values = {
        'Location': request.form['Location'],
        'MinTemp': float(request.form['MinTemp']),
        'MaxTemp': max_temp,
        'Rainfall': float(request.form['Rainfall']),
        'WindGustDir': request.form['WindGustDir'],
        'WindGustSpeed': float(request.form['WindGustSpeed']),
        'WindDir9am': request.form['WindDir9am'],
        'WindDir3pm': request.form['WindDir3pm'],
        'WindSpeed9am': float(request.form['WindSpeed9am']),
        'WindSpeed3pm': float(request.form['WindSpeed3pm']),
        'Humidity9am': float(request.form['Humidity9am']),
        'Humidity3pm': float(request.form['Humidity3pm']),
        'Pressure9am': float(request.form['Pressure9am']),
        'Pressure3pm': float(request.form['Pressure3pm']),
        'Temp9am': float(request.form['Temp9am']),
        'Temp3pm': float(request.form['Temp3pm']),
        'RainToday': request.form['RainToday']
    }

    input_data = preprocess_input(raw_input_values)

    # loading the model file from the storage
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # predictions using the loaded model file
    predicted_probability = loaded_model.predict_proba(input_data)[:, 1]

    if predicted_probability >= 0.5:
        prediction = "High probability of rain"
    else:
        prediction = "Low probability of rain"

    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
