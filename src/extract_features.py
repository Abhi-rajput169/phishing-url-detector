import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse

# Load dataset headers to maintain the correct feature order
df_sample = pd.read_csv("data/dataset_phishing.csv", nrows=1)  # Load only headers
feature_columns = list(df_sample.columns)
feature_columns.remove("url")  # Remove 'url' column
feature_columns.remove("status")  # Remove target column

def extract_features(url):
    """
    Extract features from a given URL.
    :param url: The input URL.
    :return: A list of extracted features (matching training features).
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc
    path = parsed_url.path

    features = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": 1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0,
        "https": 1 if url.startswith("https") else 0,
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(c in ['-', '_', '@', '?', '%', '&', '='] for c in url),
        "num_subdomains": hostname.count('.'),
        "url_entropy": -sum(p * np.log2(p) for p in pd.Series(list(url)).value_counts(normalize=True)),
    }

    # Fill missing features with 0 (to match dataset columns)
    extracted_features = {col: features.get(col, 0) for col in feature_columns}

    return list(extracted_features.values())

if __name__ == "__main__":
    test_url = "http://example-phishing-site.com/login"
    features = extract_features(test_url)
    print("Extracted Features:", features)
    print(f"Feature count: {len(features)} (Expected: {len(feature_columns)})")
