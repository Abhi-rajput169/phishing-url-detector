import re
import whois
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import joblib
from datetime import datetime
import logging
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load feature columns from training
try:
    feature_columns = joblib.load("models/training_columns.pkl")
    logger.info(f"Loaded feature columns: {len(feature_columns)}")
except FileNotFoundError:
    logger.error("training_columns.pkl not found! Run train.py first.")
    feature_columns = []

TRUSTED_DOMAINS = {
    'openai.com', 'chatgpt.com', 'leetcode.com',
    'google.com', 'amazon.com', 'microsoft.com'
}

def is_trusted_domain(hostname):
    """Check if domain or its parent domains are trusted."""
    domain_parts = hostname.split('.')
    for i in range(len(domain_parts)):
        current_domain = '.'.join(domain_parts[i:])
        if current_domain in TRUSTED_DOMAINS:
            return 1
    return 0

def extract_features(url):
    """Extract features from URL matching training data format."""
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc
    path = parsed_url.path

    features = {
        
        # URL Structure Features
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname))),
        "nb_dots": url.count('.'),
        "nb_hyphens": url.count('-'),
        "nb_at": url.count('@'),
        "nb_qm": url.count('?'),
        "nb_and": url.count('&'),
        "nb_or": url.count('|'),
        "nb_eq": url.count('='),
        "nb_underscore": url.count('_'),
        "nb_tilde": url.count('~'),
        "nb_percent": url.count('%'),
        "nb_slash": url.count('/'),
        "nb_star": url.count('*'),
        "nb_colon": url.count(':'),
        "nb_comma": url.count(','),
        "nb_semicolumn": url.count(';'),
        "nb_dollar": url.count('$'),
        "nb_space": url.count(' '),
        "nb_www": int("www." in hostname),
        "nb_com": int(".com" in hostname),
        "nb_dslash": url.count('//'),
        "http_in_path": int("http" in path),
        "https_token": int("https" in url[5:]),
        
        # Character Distribution
        "ratio_digits_url": sum(c.isdigit() for c in url) / max(len(url), 1),
        "ratio_digits_host": sum(c.isdigit() for c in hostname) / max(len(hostname), 1),
        
        # Domain Analysis
        "punycode": int("xn--" in url),
        "port": int(":" in hostname),
        "tld_in_path": int(any(tld in path for tld in [".com", ".net", ".org"])),
        "tld_in_subdomain": int(any(tld in hostname for tld in [".com", ".net", ".org"])),
        "abnormal_subdomain": int(hostname.count('.') > 2),
        "nb_subdomains": hostname.count('.'),
        "prefix_suffix": int('-' in hostname),
        "random_domain": int(bool(re.search(r'\d{5,}', hostname))),
        "shortening_service": int(any(s in hostname for s in ["bit.ly", "goo.gl", "tinyurl"])),
        "typosquatting": int(re.search(r'(paypa1|g00gle|amaz0n|netfl1x)', hostname, re.IGNORECASE) is not None),
        "suspicious_subdomain": int("login" in hostname or "secure" in hostname),
        
        # Content Analysis
        "nb_hyperlinks": -1,
        "login_form": -1,
        "external_favicon": -1,
        
        # Domain Reputation
        "domain_age": -1,
        "dns_record": 0,
        "trusted_domain": is_trusted_domain(hostname),
    }

    # --- HTML-based Features ---
    try:
        response = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        features.update({
            "nb_hyperlinks": len(soup.find_all('a')),
            "login_form": int(bool(soup.find('input', {'type': 'password'}))),
            "external_favicon": int(bool(soup.find('link', rel='icon')))
        })
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")

    # --- Domain Age Calculation ---
    try:
        domain_info = whois.whois(hostname)
        if domain_info.creation_date:
            creation_date = (
                min(domain_info.creation_date) 
                if isinstance(domain_info.creation_date, list) 
                else domain_info.creation_date
            )
            if creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=None)
            features["domain_age"] = (datetime.now() - creation_date).days
    except Exception as e:
        logger.error(f"Domain age error: {e}")

    # Create DataFrame with correct columns
    df = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)
    
    # Debug feature alignment
    if len(df.columns) != len(feature_columns):
        logger.error(f"Feature mismatch! Expected {len(feature_columns)}, got {len(df.columns)}")
        raise ValueError("Feature columns don't match training data")
    
    return df

if __name__ == "__main__":
    test_url = "https://chatgpt.com"
    features_df = extract_features(test_url)
    print("\nExtracted Features:")
    print(features_df.T)