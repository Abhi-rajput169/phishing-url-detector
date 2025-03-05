import joblib
from extract_features import extract_features

# Load the trained model
model = joblib.load("models/phishing_model.pkl")

def predict_url(url):
    """
    Predict if a given URL is phishing or legitimate.
    :param url: The input URL.
    :return: "Phishing" or "Legitimate"
    """
    features = extract_features(url)
    prediction = model.predict([features])[0]
    return "Phishing" if prediction == 1 else "Legitimate"

if __name__ == "__main__":
    url = input("Enter a URL: ")
    result = predict_url(url)
    print(f"Prediction: {result}")
