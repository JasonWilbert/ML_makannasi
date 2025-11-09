import re
import string
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# === Load Model Components ===
model = joblib.load('phishing_detection_model/xgboost_phishing_model.pkl')
tfidf_vectorizer = joblib.load('phishing_detection_model/tfidf_vectorizer.pkl')
numeric_features = joblib.load('phishing_detection_model/numeric_features.pkl')
target_col = joblib.load('phishing_detection_model/target_col.pkl')


# === TEXT PREPROCESSING ===
def enhanced_preprocess_combined_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "url", text)        # Replace URLs
    text = re.sub(r"\S+@\S+", "emailaddr", text)         # Replace emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# === FEATURE EXTRACTION ===
def extract_phishing_features(text: str) -> dict:
    """Extract numeric phishing indicators from text."""
    if not isinstance(text, str):
        text = ""

    features = {}
    text = text.lower()

    # Keyword-based suspiciousness
    phishing_keywords = [
        "urgent", "verify", "password", "login", "click", "account",
        "update", "security", "confirm", "bank", "paypal", "alert"
    ]
    features["suspicious_keyword_count"] = sum(1 for word in phishing_keywords if word in text)

    # Presence of link or email
    features["has_link"] = 1 if re.search(r"http[s]?://", text) else 0
    features["has_email"] = 1 if re.search(r"\S+@\S+", text) else 0

    # Count digits (like account numbers or OTPs)
    features["digit_count"] = sum(c.isdigit() for c in text)

    # Brand mention consistency
    brands = ["paypal", "amazon", "bank", "google", "apple", "microsoft"]
    mentioned_brands = [brand for brand in brands if brand in text]
    features["sender_content_mismatch"] = 1 if len(mentioned_brands) > 1 else 0

    # ✅ NEW FEATURE: has_legitimate_short_domain
    features["has_legitimate_short_domain"] = 1 if re.search(
        r"\b(bit\.ly|t\.co|tinyurl\.com|goo\.gl|rb\.gy)\b", text
    ) else 0

    # ✅ Add fallback 0 values for any missing keys (prevents KeyError)
    for f in numeric_features:
        if f not in features:
            features[f] = 0

    return features


# === COMBINE FEATURES (TF-IDF + NUMERIC) ===
def extract_features(email_text: str):
    """Combine TF-IDF text features with numeric phishing features."""
    preprocessed = enhanced_preprocess_combined_text(email_text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed]).toarray()
    phishing_feats = extract_phishing_features(preprocessed)
    numeric_vector = np.array([phishing_feats[f] for f in numeric_features]).reshape(1, -1)
    combined_features = np.hstack((tfidf_vector, numeric_vector))
    return combined_features


# === PREDICTION ===
def predict_email_type(features, model, tfidf_vectorizer, numeric_features, target_col):
    """Predict the email type and probability."""
    pred_index = model.predict(features)[0]
    pred_label = target_col[pred_index]
    pred_prob = model.predict_proba(features).max()
    return pred_label, pred_prob


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # === Load the input CSV ===
    df = pd.read_csv('valid.csv')

    # Ensure required columns exist
    if 'Email Text' not in df.columns or 'Email Type' not in df.columns:
        raise ValueError("CSV must contain 'Email Text' and 'Email Type' columns")

    print("🔍 Predicting email types...")
    predictions = []
    probabilities = []

    # Predict for each email
    for text in tqdm(df['Email Text'], desc="Processing emails"):
        features = extract_features(text)
        pred, prob = predict_email_type(features, model, tfidf_vectorizer, numeric_features, target_col)
        predictions.append(pred)
        probabilities.append(prob)

    # Add results to dataframe
    df['Predicted Type'] = predictions
    df['Confidence'] = probabilities
    df['Confirm'] = df.apply(
        lambda row: "Yes" if str(row['Predicted Type']).lower() == str(row['Email Type']).lower() else "No",
        axis=1
    )

    # Save output
    df.to_csv('valid_confirmed.csv', index=False)
    print("✅ Done! Results saved as 'valid_confirmed.csv'")
    print(df[['Email Text', 'Email Type', 'Predicted Type', 'Confirm']].head())
