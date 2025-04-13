import joblib
import pandas as pd
from extract_features import extract_features
import numpy as np

# Load the trained model
model = joblib.load("models/phishing_model.pkl")

def predict_url(url):
    features_df = extract_features(url)
    
    # Ensure columns match training data
    features_df = features_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Fill missing values (if any)
    training_medians = joblib.load("models/training_medians.pkl")
    features_df = features_df.fillna(training_medians)
    
    prediction = model.predict(features_df)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

if __name__ == "__main__":
    url = input("Enter a URL: ")
    result = predict_url(url)
    print(f"Prediction: {result}")
