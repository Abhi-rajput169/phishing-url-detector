import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert target column ('status') to numerical values (0: legitimate, 1: phishing)
    label_encoder = LabelEncoder()
    df['status'] = label_encoder.fit_transform(df['status'])

    # Drop 'url' column since it's not useful for model training
    df = df.drop(columns=['url'])

    # Handle missing values (if any)
    df = df.fillna(df.median())

    return df

if __name__ == "__main__":
    dataset_path = "data\dataset_phishing.csv"  # Updated dataset path
    df = load_and_preprocess_data(dataset_path)
    print("Preprocessing complete. Sample data:")
    print(df.head())
