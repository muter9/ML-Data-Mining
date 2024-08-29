import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
import pickle
from car_data_prep import prepare_data

# Load the dataset
data = pd.read_csv('dataset.csv')
train_data, preprocessor, t_scaler = prepare_data(data, fit=True)

# X is the processed train_data
X = train_data.drop(columns='Price')

# Align y with the processed X
y = train_data['Price']

# Define and train the model
model = ElasticNetCV(cv=10, random_state=42)
model.fit(X, y)

# Save the trained model and preprocessor
pickle.dump(model, open('trained_model.pkl', 'wb'))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))
pickle.dump(t_scaler, open('target_scaler.pkl', 'wb'))
