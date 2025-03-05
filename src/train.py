import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/dataset_phishing.csv")

# Convert target column ('status') to numerical values (0: legitimate, 1: phishing)
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# Drop 'url' column since it's not useful for model training
df = df.drop(columns=['url'])

# Handle missing values
df = df.fillna(df.median())

# Split data into features (X) and target (y)
X = df.drop(columns=['status'])  # Features
y = df['status']  # Target variable

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the trained model
joblib.dump(model, "models/phishing_model.pkl")
print("Model saved successfully.")
