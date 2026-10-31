from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

# Flask application
app = Flask(__name__, template_folder="Templates")

# Get current project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML model
model_path = os.path.join(
    BASE_DIR, "Models", "finalized_model_RandomForest.sav"
)

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load scaler
scaler_path = os.path.join(
    BASE_DIR, "Models", "scaler.pkl"
)

with open(scaler_path, "rb") as file:
    sc = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['Age_input'])
    albumin = float(request.form['al_input'])
    bgr = float(request.form['bgr_input'])
    bu = float(request.form['bu_input'])
    sc_value = float(request.form['sc_input'])
    hemo = float(request.form['hemo_input'])
    pcv = float(request.form['pcv_input'])
    wc = float(request.form['wc_input'])

    input_data = np.array([[
        age, albumin, bgr, bu,
        sc_value, hemo, pcv, wc
    ]])

    scaled_data = sc.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 0:
        result = "CKD Detected"
    else:
        result = "Not CKD"

    return render_template(
        'result.html',
        prediction=result
    )


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )

    