import gradio as gr
import re
import numpy as np
import tensorflow as tf
import joblib
from urllib.parse import urlparse
from tld import get_tld

model = tf.keras.models.load_model("LSTM.h5")
scaler = joblib.load("scaler.pkl")   # MUST be the original scaler

def having_ip_address(url):
    return int(bool(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', url)))

def abnormal_url(url):
    hostname = urlparse(url).netloc
    return int(hostname not in url)

def suspicious_words(url):
    return int(bool(re.search(
        r'paypal|login|signin|bank|account|update|free|secure|bonus',
        url, re.IGNORECASE
    )))

def fd_length(url):
    try:
        return len(urlparse(url).path.split('/')[1])
    except:
        return 0

def tld_length(url):
    try:
        return len(get_tld(url, fail_silently=True))
    except:
        return 0

def looks_like_typosquatting(url):
    return int(bool(re.search(r'[a-z]{2,}\d+[a-z]+', url)))

def invalid_tld(url):
    try:
        url = normalize_url(url)
        get_tld(url)
        return 0
    except:
        return 1

def subdomain_count(url):
    hostname = urlparse(url).netloc
    return hostname.count('.') - 1

def normalize_url(url):
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url

BRANDS = ["google", "facebook", "icloud", "paypal", "amazon", "microsoft"]

def brand_impersonation(url):
    hostname = urlparse(url).netloc.lower()
    for brand in BRANDS:
        if brand in hostname and not hostname.startswith(brand):
            return 1
    return 0

def random_domain(url):
    hostname = urlparse(url).netloc
    return int(bool(re.search(r'[a-z]{10,}', hostname)))

def extract_features(url):
    url = normalize_url(url)
    features = [
        having_ip_address(url),
        abnormal_url(url),
        url.count('.'),
        url.count('www'),
        url.count('@'),
        urlparse(url).path.count('/'),
        urlparse(url).path.count('//'),
        int(bool(re.search(r'bit\.ly|tinyurl|t\.co', url))),
        url.count('%'),
        url.count('?'),
        url.count('-'),
        url.count('='),
        len(url),
        url.count('https'),
        url.count('http'),
        len(urlparse(url).netloc),
        suspicious_words(url),
        fd_length(url),
        tld_length(url),
        sum(c.isdigit() for c in url),
        sum(c.isalpha() for c in url)
    ]
    return np.array(features).reshape(1, -1)


def predict_url(url):
    url = normalize_url(url)

    if brand_impersonation(url):
        return "🟡 Suspicious (Brand impersonation)"

    if subdomain_count(url) >= 3:
        return "🟡 Suspicious (Excessive subdomains)"

    if random_domain(url):
        return "🟡 Suspicious (Random-looking domain)"

    # ML fallback
    features = extract_features(url)
    features_scaled = scaler.transform(features)
    features_lstm = features_scaled.reshape((1, features_scaled.shape[1], 1))

    pred = model.predict(features_lstm, verbose=0)[0][0]

    if pred > 0.6:
        return "🔴 Malicious"
    elif pred < 0.4:
        return "🟢 Benign"
    else:
        return "🟡 Suspicious"


iface = gr.Interface(
    # ADD these urls in the interface for testing:
    # nugget.ca/ArticleDisplay.aspx?archive=true&e=1160966 benign
    # br-icloud.com.br  malicious
    # http://www.pashminaonline.com/pure-pashminas malicious
    # https://chatgpt.com benign
    # http://www.marketingbyinternet.com/mo/e56508df639f6ce7d55c81ee3fcd5ba8/ malicious
    fn=predict_url,
    inputs=gr.Textbox(label="Enter URL", placeholder="https://example.com"),
    outputs=gr.Textbox(label="Prediction"),
    title="Malicious URL Detector (LSTM + Scaler)",
    description="Feature-based LSTM model for malicious URL detection."
)

iface.launch()
