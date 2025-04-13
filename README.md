# Phishing URL Detection using Machine Learning 🔐

This is a Flask-based web app that predicts whether a given URL is **phishing or legitimate** using a trained XGBoost model.

## 🚀 Features
- URL feature extraction
- Trained on 11K+ dataset
- XGBoost model with 98% accuracy
- Real-time prediction via web UI

## 🧠 Technologies Used
- Python
- Flask
- XGBoost
- Pandas, NumPy, Scikit-learn
- HTML + CSS (for frontend)
- Render.com (for deployment)

## 📂 Project Structure
```
project/
├── app.py  
├── models/  
│   └── phishing_model.pkl  
├── src/  
│   ├── extract_features.py  
│   ├── templates/  
│   │   └── index.html  
│   └── static/  
│       └── style.css  
├── requirements.txt  
└── Procfile
```

## 🛠️ How to Run Locally

```bash
git clone https://github.com/Abhi-rajput169/phishing-url-detector.git
cd phishing-url-detector
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser 🚀

## 🌐 Live Demo

👉 [Click here to try it on Render](https://your-render-link.com)

## 🙏 Acknowledgement
Thanks to my mentor **Mr. Gazy Abbas** and the NIMS University CSE department.

## 📬 Contact
Abhishek Rajput  
[GitHub](https://github.com/Abhi-rajput169) | Email: your-email@example.com
