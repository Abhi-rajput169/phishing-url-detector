import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# --- Update Dataset with New Features and URLs ---
def update_dataset():
    """Add new features and URLs to the dataset."""
    try:
        # Load the dataset
        df = pd.read_csv("data/dataset_phishing.csv")

        # Add new features (initialize with default values)
        new_features = ['ssl_valid', 'dns_record', 'trusted_domain', 'has_favicon']
        for feature in new_features:
            if feature not in df.columns:
                df[feature] = 0  # Default value

        # Add new legitimate URLs
        new_urls = [
            {"url": "https://chat.openai.com", "status": "legitimate", "ssl_valid": 1, "dns_record": 1, "trusted_domain": 1, "has_favicon": 1},
            {"url": "https://www.amazon.com", "status": "legitimate", "ssl_valid": 1, "dns_record": 1, "trusted_domain": 1, "has_favicon": 1},
            # Add more URLs as needed
        ]

        # Append new URLs to the dataset
        new_df = pd.DataFrame(new_urls)
        df = pd.concat([df, new_df], ignore_index=True)

        # Save the updated dataset
        df.to_csv("data/dataset_phishing.csv", index=False)
        print("Dataset updated successfully!")
    except Exception as e:
        print(f"Error updating dataset: {e}")

# --- Main Training Function ---
def train_model():
    """Train the phishing detection model."""
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

    # --- Save training medians and feature names ---
    os.makedirs("models", exist_ok=True)
    training_medians = X_train.median()
    joblib.dump(training_medians, "models/training_medians.pkl")
    joblib.dump(X_train.columns, "models/training_columns.pkl")

    # Define models to test
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    # Train and evaluate models
    xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
    }
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Hyperparameter tuning for XGBoost
        if name == "XGBoost":
            print("Performing hyperparameter tuning for XGBoost...")
            grid_search = GridSearchCV(model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best XGBoost parameters: {grid_search.best_params_}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print(f"{name} Test Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

        # Print feature importance if applicable
        if name in ["Decision Tree", "Random Forest", "XGBoost"]:
            feature_importance = model.feature_importances_
            important_features = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top 10 Important Features for {name}:")
            for feature, importance in important_features:
                print(f"  {feature}: {importance:.4f}")

    # Save the best model
    joblib.dump(best_model, "models/phishing_model.pkl")
    print(f"\n Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f}) saved successfully!")

    print("\nTraining features:", X_train.columns.tolist())

# --- Main Execution ---
if __name__ == "__main__":
    # Update dataset with new features and URLs
    update_dataset()
    
    # Train the model
    train_model()