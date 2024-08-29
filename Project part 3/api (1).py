# In your api.py

from flask import Flask, request, render_template
import pickle
import pandas as pd
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the model and preprocessor
model = pickle.load(open('trained_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
t_scaler = pickle.load(open('target_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [x for x in request.form.values()]
    data = pd.DataFrame([features], columns=['manufactor', 'model', 'Year', 'Km', 'Hand', 'Engine_type', 'capacity_Engine', 'Cre_date', 'Repub_date', 'Gear'])

    # Preprocess the data using the loaded preprocessor
    processed_data = prepare_data(data, fit = False, preprocessor = preprocessor)
    
    # Predict using the model
    prediction = model.predict(processed_data)
    prediction = t_scaler.inverse_transform(prediction.reshape(-1, 1))
    prediction = prediction[0][0]
    if prediction < 10000:
    	prediction = 9999.99
    
    return render_template('index.html', prediction_text='Predicted Car Price: ₪{:.2f}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
