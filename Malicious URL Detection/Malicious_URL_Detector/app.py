# ========================= First Version =========================
# import gradio as gr
# import re
# from urllib.parse import urlparse
# from tld import get_tld
# from googlesearch import search
# import numpy as np
# import tensorflow as tf

# import re
# from urllib.parse import urlparse
# from tld import get_tld
# from googlesearch import search

# def having_ip_address(url):
#     match = re.search(
#         r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])|'
#         r'((0x[0-9a-fA-F]{1,2})\.){3}(0x[0-9a-fA-F]{1,2})|'
#         r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
#     return 1 if match else 0

# def abnormal_url(url):
#     hostname = str(urlparse(url).hostname)
#     return 1 if hostname and hostname in url else 0

# def google_index(url):
#     try:
#         site = search(url, num_results=5)
#         return 1 if site else 0
#     except:
#         return 0

# def count_dot(url): return url.count('.')
# def count_www(url): return url.count('www')
# def count_atrate(url): return url.count('@')
# def no_of_dir(url): return urlparse(url).path.count('/')
# def no_of_embed(url): return urlparse(url).path.count('//')

# def suspicious_words(url):
#     return 1 if re.search(r'paypal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url, re.IGNORECASE) else 0

# def shortening_service(url):
#     return 1 if re.search(
#         r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
#         r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
#         r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
#         r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
#         r'db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|'
#         r'twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|'
#         r'scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net',
#         url) else 0

# def count_https(url): return url.count('https')
# def count_http(url): return url.count('http')
# def count_per(url): return url.count('%')
# def count_ques(url): return url.count('?')
# def count_hyphen(url): return url.count('-')
# def count_equal(url): return url.count('=')
# def url_length(url): return len(str(url))
# def hostname_length(url): return len(urlparse(url).netloc)

# def fd_length(url):
#     try:
#         return len(urlparse(url).path.split('/')[1])
#     except:
#         return 0

# def tld_length(tld):
#     try:
#         return len(tld)
#     except:
#         return -1

# def digit_count(url): return sum(c.isdigit() for c in url)
# def letter_count(url): return sum(c.isalpha() for c in url)

# # Full feature extractor for a single URL
# def extract_features(url, include_google_index=False):
#     features = [
#         having_ip_address(url),
#         abnormal_url(url),
#         count_dot(url),
#         count_www(url),
#         count_atrate(url),
#         no_of_dir(url),
#         no_of_embed(url),
#         shortening_service(url),
#         count_per(url),
#         count_ques(url),
#         count_hyphen(url),
#         count_equal(url),
#         url_length(url),
#         count_https(url),
#         count_http(url),
#         hostname_length(url),
#         suspicious_words(url),
#         fd_length(url),
#         tld_length(get_tld(url, fail_silently=True)),
#         digit_count(url),
#         letter_count(url)
#     ]
#     if include_google_index:
#         features.insert(2, google_index(url))  # Insert after abnormal_url
#     return features


# # Load your trained model
# model = tf.keras.models.load_model("LSTM_enhanced.h5")
# # Feature extraction wrapper
# def extract_features(url):
#     features = [
#         having_ip_address(url),
#         abnormal_url(url),
#         count_dot(url),
#         count_www(url),
#         count_atrate(url),
#         no_of_dir(url),
#         no_of_embed(url),
#         shortening_service(url),
#         count_per(url),
#         count_ques(url),
#         count_hyphen(url),
#         count_equal(url),
#         url_length(url),
#         count_https(url),
#         count_http(url),
#         hostname_length(url),
#         suspicious_words(url),
#         fd_length(url),
#         tld_length(get_tld(url, fail_silently=True)),
#         digit_count(url),
#         letter_count(url)
#     ]
#     return np.array([features])

# # Prediction function
# def predict_url(url):
#     features = extract_features(url)
#     pred = model.predict(features)[0][0]  # binary classification with sigmoid
#     print(pred)
#     label = "Bengin" if pred > 0.85 else "Malicious"
#     return f"Prediction: {label} )"


# iface = gr.Interface(
#     fn=predict_url,
#     inputs=gr.Textbox(label="Enter URL"),
#     outputs=gr.Textbox(label="Prediction"),
#     title="Malicious URL Detector",
#     description="Paste a URL to check if it's potentially malicious based on URL features."
# )

# iface.launch()


# ========================= Second Version =========================
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

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
        return "🟡 Suspicious"

    if subdomain_count(url) >= 3:
        return "🟡 Suspicious"

    if random_domain(url):
        return "🟡 Suspicious"

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



with gr.Blocks() as iface:
    gr.Markdown("### Examples to test")
    gr.Markdown(
        "- `nugget.ca/ArticleDisplay.aspx?archive=true` (benign)\n"
        "- `br-icloud.com.br` (malicious)\n"
        "- `http://www.pashminaonline.com/pure-pashminas` (malicious)\n"
        "- `https://chatgpt.com` (benign)\n"
        "- `http://www.marketingbyinternet.com/mo/e56508df639f6ce7d55c81ee3fcd5ba8/` (malicious)"
    )

    gr.Interface(
        fn=predict_url,
        inputs=gr.Textbox(label="Enter URL", placeholder="https://example.com", lines=1, max_lines=1),
        outputs=gr.Textbox(label="Prediction", lines=1, max_lines=1),
        title="Malicious URL Detector (LSTM + Scaler)",
        description="Feature-based LSTM model for malicious URL detection.",

        examples=[
            ["nugget.ca/ArticleDisplay.aspx?archive=true&e=1160966"],  # benign
            ["br-icloud.com.br"],                                     # malicious
            ["http://www.pashminaonline.com/pure-pashminas"],         # malicious
            ["https://chatgpt.com"],                                  # benign
            ["https://google.com"],  
            ["http://www.designeremdoces.com/components/com_contact/ggdrives/"],  # malicious
            ["facebook.com/opalhilldrive"],  # benign
            ["citiprepaid-salarysea-at.tk"],                                 # malicious
            ["facebook.com"],                                  # benign
            ["telegram.org"],                                  # benign
            ["http://www.marketingbyinternet.com/mo/e56508df639f6ce7d55c81ee3fcd5ba8/"]  # malicious
        ],examples_per_page=11
    )

iface.launch()

