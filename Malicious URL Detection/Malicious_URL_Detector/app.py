import gradio as gr
import re
from urllib.parse import urlparse
from tld import get_tld
from googlesearch import search
import numpy as np
import tensorflow as tf

import re
from urllib.parse import urlparse
from tld import get_tld
from googlesearch import search

def having_ip_address(url):
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])|'
        r'((0x[0-9a-fA-F]{1,2})\.){3}(0x[0-9a-fA-F]{1,2})|'
        r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    return 1 if match else 0

def abnormal_url(url):
    hostname = str(urlparse(url).hostname)
    return 1 if hostname and hostname in url else 0

def google_index(url):
    try:
        site = search(url, num_results=5)
        return 1 if site else 0
    except:
        return 0

def count_dot(url): return url.count('.')
def count_www(url): return url.count('www')
def count_atrate(url): return url.count('@')
def no_of_dir(url): return urlparse(url).path.count('/')
def no_of_embed(url): return urlparse(url).path.count('//')

def suspicious_words(url):
    return 1 if re.search(r'paypal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url, re.IGNORECASE) else 0

def shortening_service(url):
    return 1 if re.search(
        r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
        r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
        r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
        r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
        r'db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|'
        r'twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|'
        r'scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net',
        url) else 0

def count_https(url): return url.count('https')
def count_http(url): return url.count('http')
def count_per(url): return url.count('%')
def count_ques(url): return url.count('?')
def count_hyphen(url): return url.count('-')
def count_equal(url): return url.count('=')
def url_length(url): return len(str(url))
def hostname_length(url): return len(urlparse(url).netloc)

def fd_length(url):
    try:
        return len(urlparse(url).path.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def digit_count(url): return sum(c.isdigit() for c in url)
def letter_count(url): return sum(c.isalpha() for c in url)

# Full feature extractor for a single URL
def extract_features(url, include_google_index=False):
    features = [
        having_ip_address(url),
        abnormal_url(url),
        count_dot(url),
        count_www(url),
        count_atrate(url),
        no_of_dir(url),
        no_of_embed(url),
        shortening_service(url),
        count_per(url),
        count_ques(url),
        count_hyphen(url),
        count_equal(url),
        url_length(url),
        count_https(url),
        count_http(url),
        hostname_length(url),
        suspicious_words(url),
        fd_length(url),
        tld_length(get_tld(url, fail_silently=True)),
        digit_count(url),
        letter_count(url)
    ]
    if include_google_index:
        features.insert(2, google_index(url))  # Insert after abnormal_url
    return features


# Load your trained model
model = tf.keras.models.load_model("LSTM_enhanced.h5")
# Feature extraction wrapper
def extract_features(url):
    features = [
        having_ip_address(url),
        abnormal_url(url),
        count_dot(url),
        count_www(url),
        count_atrate(url),
        no_of_dir(url),
        no_of_embed(url),
        shortening_service(url),
        count_per(url),
        count_ques(url),
        count_hyphen(url),
        count_equal(url),
        url_length(url),
        count_https(url),
        count_http(url),
        hostname_length(url),
        suspicious_words(url),
        fd_length(url),
        tld_length(get_tld(url, fail_silently=True)),
        digit_count(url),
        letter_count(url)
    ]
    return np.array([features])

# Prediction function
def predict_url(url):
    features = extract_features(url)
    pred = model.predict(features)[0][0]  # binary classification with sigmoid
    print(pred)
    label = "Bengin" if pred > 0.85 else "Malicious"
    return f"Prediction: {label} )"


iface = gr.Interface(
    fn=predict_url,
    inputs=gr.Textbox(label="Enter URL"),
    outputs=gr.Textbox(label="Prediction"),
    title="Malicious URL Detector",
    description="Paste a URL to check if it's potentially malicious based on URL features."
)

iface.launch()
