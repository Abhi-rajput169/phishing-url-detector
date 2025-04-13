from flask import Flask, request, render_template
import joblib
import logging
from src.extract_features import extract_features
import os

# Configure Flask app
app = Flask(__name__, 
            template_folder="src/templates",  # Path to templates folder
            static_folder="src/static")       # Path to static folder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
def load_model():
    try:
        model = joblib.load("models/phishing_model.pkl")
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(" Model file not found! Run train.py first.")
        return None

model = load_model()

def validate_url(url):
    """Validate URL format"""
    return url.startswith(('http://', 'https://')) if url else False

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        
        if not validate_url(url):
            return render_template("index.html", 
                                error="Invalid URL format. Must start with http:// or https://",
                                url=url)
            
        try:
            if not model:
                raise ValueError("Model not loaded - check server logs")
            
            features_df = extract_features(url)
            
            if features_df.empty:
                raise ValueError("Feature extraction failed")
                
            features = features_df.values.flatten()
            
            # Get prediction and confidence
            prediction = model.predict([features])[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([features])[0]
                confidence = round(max(proba) * 100, 2)
            
            result = "Phishing" if prediction == 1 else "Legitimate"
            
            # Post-prediction override for trusted domains
            if features_df["trusted_domain"].values[0] == 1:
                result = "Legitimate"
                confidence = 99.99  # Force high confidence
            
            return render_template("index.html", 
                                url=url, 
                                result=result,
                                confidence=confidence)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template("index.html", 
                                error=f"Error processing URL: {e}",
                                url=url)
    
    return render_template("index.html")

if __name__ == "__main__":
    if not os.path.exists("models/phishing_model.pkl"):
        logger.critical("Missing model file! Run train.py first.")
    app.run(host='0.0.0.0', port=5000, debug=True)