import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd # Changed pf to pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Import ridge regressor and standard scaler pickle
# Ensure these file paths are correct relative to app.py
ridge_model = pickle.load(open('models/ridge.pkl', "rb"))
standard_scaler = pickle.load(open('models/scalar.pkl', "rb")) # Fixed spelling from 'standar'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # 1. Extract all features correctly
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC')) # Added this
        ISI = float(request.form.get('ISI')) # Fixed the mapping here
        Classes = float(request.form.get('classes')) # Note: HTML 'name' was lowercase 'classes'
        Region = float(request.form.get('Region'))
        
        # 2. Scale the data (Ensure the order matches your training data!)
        # Order should be: [Temp, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # 3. Predict
        result = ridge_model.predict(new_data_scaled)

        # 4. Return result (using result[0] because predict returns a list)
        return render_template('home.html', results=result[0])
    else:
        # For GET requests, just show the form
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)