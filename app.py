from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack, csr_matrix
import os
from datetime import datetime
import nltk
from nltk.corpus import stopwords

# Download stopwords jika belum ada
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model dan komponen yang diperlukan
model_dir = 'phishing_detection_model'
model = joblib.load(os.path.join(model_dir, 'xgboost_phishing_model.pkl'))
tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
numeric_features = joblib.load(os.path.join(model_dir, 'numeric_features.pkl'))
target_col = joblib.load(os.path.join(model_dir, 'target_col.pkl'))
model_metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))

# Definisikan fungsi preprocessing dan ekstraksi fitur yang sama dengan notebook
def enhanced_preprocess_combined_text(text):
    # Ekstraksi komponen penting
    date_pattern = r'\w{3}\s\w{3}\s\d{1,2}\s\d{4}'
    sender_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'

    # Ekstraksi tanggal
    date_match = re.search(date_pattern, text)
    extracted_date = date_match.group(0) if date_match else ""

    # Ekstraksi pengirim
    sender_match = re.search(sender_pattern, text)
    extracted_sender = sender_match.group(0) if sender_match else ""

    # Bersihkan teks dengan penanganan khusus untuk phishing
    clean_text = re.sub(date_pattern, '', text)
    clean_text = re.sub(sender_pattern, '', clean_text)

    # Normalisasi karakter evasi
    clean_text = clean_text.replace('â€', "'")
    clean_text = clean_text.replace('â€œ', '"')
    clean_text = clean_text.replace('â€˜', "'")

    # Hapus pola attachment
    clean_text = re.sub(r'see attached file', '', clean_text, flags=re.IGNORECASE)

    # Hapus karakter khusus tapi pertahankan tanda baca penting
    clean_text = re.sub(r'[^\w\s\.\!\?]', '', clean_text)

    # Lowercase
    clean_text = clean_text.lower()

    # Hapus stopwords
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join([word for word in clean_text.split() if word not in stop_words])

    return clean_text, extracted_date, extracted_sender

def extract_phishing_features(text):
    features = {}

    # ... (kode di sini sama dan tidak berubah) ...
    # 1. Suspicious Keywords (diperluas)
    phishing_keywords = [
        'urgent', 'immediate', 'action required', 'verify your account',
        'suspended', 'limited time', 'click here', 'update now',
        'confirm', 'security alert', 'unusual sign-in', 'locked account',
        'billing issue', 'payment failed', 'account locked', 'verify identity',
        'secure your account', 'unauthorized access', 'expiring today',
        'act now', 'limited offer', 'exclusive deal', 'confirm immediately'
    ]
    features['suspicious_keyword_count'] = sum(1 for keyword in phishing_keywords if keyword in text.lower())

    # 2. Realistic Suspicious Domains
    legitimate_short_domains = ['bit.ly', 't.co', 'goo.gl', 'ow.ly', 'buff.ly', 'mcaf.ee']
    suspicious_short_domains = [
        'tinyurl.com', 'short.url', 'tiny.cc', 'is.gd', 'adf.ly',
        'vzturl.com', 'cli.re', 'q.gs', 'u.to', 'yourl.io', 'po.st'
    ]

    features['has_legitimate_short_domain'] = 1 if any(domain in text.lower() for domain in legitimate_short_domains) else 0
    features['has_suspicious_short_domain'] = 1 if any(domain in text.lower() for domain in suspicious_short_domains) else 0

    # Deteksi typosquatting
    legitimate_domains = ['paypal.com', 'amazon.com', 'microsoft.com', 'apple.com', 'google.com']
    features['has_typosquatting'] = 0

    for domain in legitimate_domains:
        if domain in text.lower():
            typo_variations = [
                domain.replace('.com', '.co'),
                domain.replace('.com', '.org'),
                domain.replace('a', '4'), domain.replace('i', '1'),
                domain.replace('o', '0'), domain.replace('l', '1'),
                domain.replace('m', 'rn'), domain.replace('n', 'rn')
            ]
            if any(typo in text.lower() for typo in typo_variations):
                features['has_typosquatting'] = 1
                break

    # Deteksi IP address sebagai URL
    ip_pattern = r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    features['has_ip_url'] = 1 if re.search(ip_pattern, text) else 0

    # 3. Advanced Capital Word Analysis
    words = text.split()
    if words:
        capital_words = [word for word in words if word.isupper() and len(word) > 1]
        features['capital_word_ratio'] = len(capital_words) / len(words)

        sentences = re.split(r'[.!?]+', text)
        all_caps_sentences = sum(1 for sentence in sentences if sentence.strip() and sentence.strip().isupper())
        features['all_caps_sentences_ratio'] = all_caps_sentences / max(len(sentences), 1)

        # Deteksi kapital tidak wajar
        brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook']
        acronyms = ['ID', 'URL', 'HTML', 'PDF', 'CEO', 'CFO']
        unusual_capitals = 0

        for i, word in enumerate(words):
            if word.isupper() and len(word) > 1:
                if i == 0 or words[i-1].endswith(('.', '!', '?')):
                    continue
                if word.lower() in brands or word in acronyms:
                    continue
                unusual_capitals += 1

        features['unusual_capital_ratio'] = unusual_capitals / len(words)
    else:
        features['capital_word_ratio'] = 0
        features['all_caps_sentences_ratio'] = 0
        features['unusual_capital_ratio'] = 0

    # 4. Advanced Exclamation Analysis
    exclamation_count = text.count('!')
    features['exclamation_count'] = exclamation_count
    features['exclamation_ratio'] = exclamation_count / max(len(text), 1)
    features['excessive_exclamation'] = 1 if exclamation_count > 5 else 0

    consecutive_exclamation = len(re.findall(r'!{3,}', text))
    features['consecutive_exclamation'] = consecutive_exclamation

    exclamation_positions = [i for i, char in enumerate(text) if char == '!']
    if exclamation_positions:
        mid_sentence_exclamations = 0
        for pos in exclamation_positions:
            if pos + 1 < len(text) and text[pos+1] not in ['.', ' ', '\n']:
                mid_sentence_exclamations += 1
        features['mid_sentence_exclamation_ratio'] = mid_sentence_exclamations / len(exclamation_positions)
    else:
        features['mid_sentence_exclamation_ratio'] = 0

    # 5. Urgency & Time Pressure
    urgency_words = ['urgent', 'immediately', 'asap', 'hurry', 'fast', 'quick', 'now', 'today', 'soon']
    features['urgency_word_count'] = sum(1 for word in urgency_words if word in text.lower())

    time_limit_keywords = ['24 hours', '48 hours', 'by tomorrow', 'today only', 'expires today']
    features['has_time_limit'] = 1 if any(keyword in text.lower() for keyword in time_limit_keywords) else 0

    # 6. Personal Information Request
    personal_info_keywords = [
        'ssn', 'social security', 'credit card', 'bank account',
        'password', 'pin', 'cvv', 'account number', 'card number',
        'expiration date', 'security code', 'routing number'
    ]
    features['personal_info_request'] = 1 if any(keyword in text.lower() for keyword in personal_info_keywords) else 0

    # 7. Threatening Language
    threat_keywords = ['suspend', 'terminate', 'close', 'deactivate', 'block', 'restrict', 'penalty', 'fee', 'fine']
    features['has_threat'] = 1 if any(keyword in text.lower() for keyword in threat_keywords) else 0

    # 8. Generic & Personalization Analysis
    generic_greetings = ['dear customer', 'dear user', 'dear sir/madam', 'valued customer', 'account holder']
    features['has_generic_greeting'] = 1 if any(greeting in text.lower() for greeting in generic_greetings) else 0

    personalization_placeholders = ['[name]', '[email]', '[customer]', '[user]']
    features['has_personalization_placeholder'] = 1 if any(ph in text.lower() for ph in personalization_placeholders) else 0

    # 9. Brand Mentions Analysis
    brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook', 'instagram']
    brand_mentions = [brand for brand in brands if brand in text.lower()]
    features['brand_mention_count'] = len(brand_mentions)

    features['inconsistent_brand_mention'] = 0
    if brand_mentions:
        for brand in brand_mentions:
            suspicious_variations = [
                brand + 'support', brand + 'security', brand + 'team',
                brand + 'update', brand + 'alert', brand + 'notice'
            ]
            if any(variation in text.lower() for variation in suspicious_variations):
                features['inconsistent_brand_mention'] = 1
                break

    # 10. Spelling Errors
    common_misspellings = {
        'paypaI': 'paypal', 'appIe': 'apple', 'microsft': 'microsoft',
        'amaz0n': 'amazon', 'g00gle': 'google', 'faceb00k': 'facebook',
        'verifye': 'verify', 'securty': 'security', 'acount': 'account'
    }
    misspelling_count = sum(1 for misspelling in common_misspellings if misspelling in text.lower())
    features['spelling_errors_count'] = misspelling_count

    # 11. HTML & Technical Content
    html_tags = ['<html', '<div', '<table', '<form', '<script', '<iframe']
    features['has_html_content'] = 1 if any(tag in text.lower() for tag in html_tags) else 0

    features['has_form_submission'] = 1 if '<form' in text.lower() and 'action=' in text.lower() else 0
    features['has_javascript'] = 1 if 'javascript:' in text.lower() or '<script' in text.lower() else 0
    features['has_tracking_pixel'] = 1 if any(pixel in text.lower() for pixel in [
        'tracking pixel', 'open tracking', 'read receipt'
    ]) else 0
    features['has_unsubscribe_link'] = 1 if 'unsubscribe' in text.lower() else 0

    # 12. Link Analysis
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    features['url_count'] = len(urls)

    misleading_anchors = ['click here', 'verify now', 'update account', 'sign in']
    features['has_misleading_link'] = 0

    for anchor in misleading_anchors:
        if anchor in text.lower():
            anchor_pos = text.lower().find(anchor)
            text_after_anchor = text[anchor_pos + len(anchor):anchor_pos + len(anchor) + 100]
            if re.search(url_pattern, text_after_anchor):
                features['has_misleading_link'] = 1
                break

    # 13. Behavioral Analysis
    features['multiple_redirects'] = 1 if len(urls) > 3 else 0
    features['shortened_url_only'] = 1 if (
        bool(re.search(r'\b(bit\.ly|t\.co|goo\.gl)\b', text)) and len(urls) == 1
    ) else 0
    features['image_only_text'] = 1 if (
        len(re.findall(r'\.(jpg|jpeg|png|gif)', text.lower())) > 0 and len(text.split()) < 20
    ) else 0

    # 14. Social Engineering Analysis
    features['authority_impersonation'] = 1 if any(impersonation in text.lower() for impersonation in [
        'fbi', 'cia', 'irs', 'police', 'government', 'bank', 'court'
    ]) else 0

    features['scarcity_tactic'] = 1 if any(scarcity in text.lower() for scarcity in [
        'only 2 left', 'last chance', 'almost gone', 'running out'
    ]) else 0

    features['social_proof'] = 1 if any(proof in text.lower() for proof in [
        'trusted by millions', 'used by fortune 500', 'recommended by experts'
    ]) else 0

    # 15. Psychological Triggers
    fear_words = ['hack', 'breach', 'compromised', 'stolen', 'fraud', 'suspended']
    features['fear_intensity'] = sum(1 for fear in fear_words if fear in text.lower())

    greed_words = ['free', 'win', 'prize', 'reward', 'discount', 'bonus']
    features['greed_trigger'] = sum(1 for greed in greed_words if greed in text.lower())

    curiosity_words = ['see what happened', 'you won\'t believe', 'shocking discovery']
    features['curiosity_trigger'] = sum(1 for curiosity in curiosity_words if curiosity in text.lower())

    # 16. Action Requests
    action_keywords = ['click', 'verify', 'update', 'confirm', 'sign in', 'log in', 'download']
    features['action_request_count'] = sum(1 for keyword in action_keywords if keyword in text.lower())

    # 17. Security Claims
    security_claims = ['secure', 'encrypted', 'protected', 'safe', 'trusted']
    features['security_claim_count'] = sum(1 for claim in security_claims if claim in text.lower())

    # 18. Attachment Analysis
    suspicious_extensions = ['.exe', '.zip', '.scr', '.bat', '.js', '.docm']
    features['has_suspicious_attachment'] = 1 if any(ext in text.lower() for ext in suspicious_extensions) else 0

    # 19. Contact Information
    suspicious_contact = [
        'call now', 'contact immediately', 'urgent call', 'phone verification',
        'verify by phone', 'confirm by call'
    ]
    features['has_suspicious_contact'] = 1 if any(contact in text.lower() for contact in suspicious_contact) else 0

    # 20. Contextual Analysis
    features['mentions_recent_events'] = 1 if any(event in text.lower() for event in [
        'covid', 'pandemic', 'election', 'holiday', 'black friday'
    ]) else 0

    features['seasonal_reference'] = 1 if any(season in text.lower() for season in [
        'christmas', 'thanksgiving', 'new year', 'summer', 'winter'
    ]) else 0

    # 21. Sender Analysis
    # <--- PERBAIKAN 1: BUG LOGIKA DI SINI ---
    # Kode lama: if 'paypal' in text.lower() and 'paypal' not in text.lower():
    # Ini akan selalu bernilai False. Seharusnya membandingkan konten dengan domain pengirim.
    # Karena fitur ini lebih tentang analisis pengirim, dan kita sudah punya `advanced_sender_analysis`,
    # fitur ini mungkin redundan atau butuh konteks sender_email. Untuk sekarang, kita set ke 0.
    # Atau, jika ingin memeriksa inkonsistensi merek dalam teks saja:
    mentioned_brands = [brand for brand in brands if brand in text.lower()]
    if len(mentioned_brands) > 1:
        features['sender_content_mismatch'] = 1 # Jika lebih dari satu merek disebut, bisa jadi mencurigakan
    else:
        features['sender_content_mismatch'] = 0

    return features

# ... (fungsi extract_url_features, extract_brand_features, dll. tidak berubah) ...
def extract_url_features(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)

    features = {}
    features['url_count'] = len(urls)

    if urls:
        features['has_url_masking'] = 1 if any('bit.ly' in url or 'tinyurl' in url for url in urls) else 0
        features['has_homograph'] = 1 if any(
            re.search(r'[āàáâãäåæçćčđēėęěğįıñňöőŕřśšşťțůűųźžż]', url)
            for url in urls
        ) else 0
    else:
        features['has_url_masking'] = 0
        features['has_homograph'] = 0

    return features

def extract_brand_features(text):
    brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook', 'instagram']
    features = {}

    for brand in brands:
        features[f'has_{brand}'] = 1 if brand in text.lower() else 0

    brand_spoofing = [
        'paypaI', 'arnazon', 'microsft', 'appIe', 'goggle',
        'faceboook', 'instagrarn'
    ]
    features['has_brand_spoofing'] = 1 if any(spoof in text.lower() for spoof in brand_spoofing) else 0

    return features

def extract_sender_features(sender_email):
    features = {}

    if '@' not in sender_email:
        features['sender_domain'] = 'unknown'
        features['is_free_email'] = 0
        features['is_legitimate_domain'] = 0
        features['is_new_domain'] = 0
        features['domain_age_days'] = -1
        return features

    # Ekstrak domain
    domain = sender_email.split('@')[-1].lower().strip()
    features['sender_domain'] = domain

    # Cek apakah domain adalah email gratis
    free_email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                         'aol.com', 'icloud.com', 'protonmail.com', 'zoho.com']
    features['is_free_email'] = 1 if domain in free_email_domains else 0

    # Daftar domain resmi perusahaan besar (bisa diperluas)
    legitimate_domains = [
        'paypal.com', 'amazon.com', 'microsoft.com', 'apple.com', 'google.com',
        'facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com',
        'ebay.com', 'netflix.com', 'spotify.com', 'adobe.com',
        'dropbox.com', 'slack.com', 'zoom.us', 'salesforce.com'
    ]
    features['is_legitimate_domain'] = 1 if domain in legitimate_domains else 0

    # Deteksi domain yang baru dibuat (kurang dari 6 bulan)
    new_domain_indicators = [
        '.tk', '.ml', '.ga', '.cf', '.gq',  # TLD gratis yang sering disalahgunakan
        '-shop', '-store', '-service', '-secure',  # Kata kunci domain mencurigakan
        'shop-', 'store-', 'service-', 'secure-'
    ]
    features['is_new_domain'] = 1 if (
        any(tld in domain for tld in ['.tk', '.ml', '.ga', '.cf', '.gq']) or
        any(indicator in domain for indicator in new_domain_indicators)
    ) else 0

    # Simulasi umur domain (dalam hari)
    if features['is_new_domain']:
        features['domain_age_days'] = 30  # Simulasi domain baru (30 hari)
    elif features['is_legitimate_domain']:
        features['domain_age_days'] = 3650  # Simulasi domain lama (10 tahun)
    else:
        features['domain_age_days'] = 365  # Simulasi domain menengah (1 tahun)

    return features

def extract_file_extension_features(text):
    features = {}

    # Daftar ekstensi file yang mencurigakan
    suspicious_extensions = {
        # Eksekusi
        'executable': ['.exe', '.scr', '.bat', '.com', '.pif', '.cmd', '.msi', '.jar'],
        # Script
        'script': ['.js', '.vbs', '.ps1', '.py', '.pl', '.rb', '.php', '.asp', '.jsp'],
        # Macro
        'macro': ['.docm', '.xlsm', '.pptm', '.dotm', '.xltm', '.potm'],
        # Arsip
        'archive': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
        # Sistem
        'system': ['.dll', '.sys', '.drv', '.ocx', '.cpl', '.deb', '.rpm'],
        # Lainnya
        'other': ['.reg', '.inf', '.iso', '.dmg', '.app', '.apk', '.deb']
    }

    # Deteksi ekstensi file dalam teks
    detected_extensions = []
    for category, extensions in suspicious_extensions.items():
        for ext in extensions:
            if ext.lower() in text.lower():
                detected_extensions.append(ext)

    # Fitur dasar
    features['has_suspicious_extension'] = 1 if detected_extensions else 0
    features['suspicious_extension_count'] = len(detected_extensions)

    # Fitur kategori
    for category, extensions in suspicious_extensions.items():
        features[f'has_{category}_extension'] = 1 if any(ext in text.lower() for ext in extensions) else 0

    # Fitur tingkat bahaya
    high_risk_extensions = ['.exe', '.scr', '.bat', '.js', '.docm', '.xlsm']
    features['has_high_risk_extension'] = 1 if any(ext in text.lower() for ext in high_risk_extensions) else 0

    # Deteksi multiple ekstensi (misal: file.exe.zip)
    multiple_ext_pattern = r'\.\w+\.\w+'
    features['has_multiple_extensions'] = 1 if re.search(multiple_ext_pattern, text.lower()) else 0

    # Deteksi ekstensi tersembunyi (misal: file.jpg.exe)
    hidden_ext_pattern = r'\.(jpg|jpeg|png|gif|pdf|txt|doc|xls)\.(exe|scr|bat|js)'
    features['has_hidden_extension'] = 1 if re.search(hidden_ext_pattern, text.lower()) else 0

    # Deteksi ekstensi yang disamarkan
    disguised_ext_patterns = [
        r'\.ex[e3]',  # exe, ex3
        r'\.sc[r7]',  # scr, sc7
        r'\.ba[t2]',  # bat, ba2
        r'\.js[a-z0-9]'  # jsa, js1, js2, dll.
    ]
    features['has_disguised_extension'] = 1 if any(re.search(pattern, text.lower()) for pattern in disguised_ext_patterns) else 0

    return features

def advanced_sender_analysis(sender_email, text_content):
    features = {}
    legitimate_domains = [
        'paypal.com', 'amazon.com', 'microsoft.com', 'apple.com', 'google.com',
        'facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com',
        'ebay.com', 'netflix.com', 'spotify.com', 'adobe.com',
        'dropbox.com', 'slack.com', 'zoom.us', 'salesforce.com'
    ]

    # Deteksi ketidaksesuaian antara pengirim dan konten
    brand_sender_mapping = {
        'paypal.com': ['paypal', 'ebay'],
        'amazon.com': ['amazon', 'aws'],
        'microsoft.com': ['microsoft', 'windows', 'office', 'outlook'],
        'apple.com': ['apple', 'icloud', 'itunes', 'iphone'],
        'google.com': ['google', 'gmail', 'youtube', 'android'],
        'facebook.com': ['facebook', 'instagram', 'whatsapp'],
        'twitter.com': ['twitter', 'tweet'],
        'linkedin.com': ['linkedin']
    }

    # Ekstrak domain pengirim
    if '@' in sender_email:
        sender_domain = sender_email.split('@')[-1].lower()

        # Cek apakah konten menyebut merek yang tidak sesuai dengan domain
        features['sender_content_mismatch'] = 0
        for domain, brands in brand_sender_mapping.items():
            if sender_domain == domain:
                # Cek apakah ada merek pesaing yang disebut
                competitor_brands = [
                    'paypal', 'amazon', 'microsoft', 'apple', 'google',
                    'facebook', 'twitter', 'linkedin'
                ]
                for brand in competitor_brands:
                    if brand in text_content.lower() and brand not in brands:
                        features['sender_content_mismatch'] = 1
                        break
                if features['sender_content_mismatch'] == 1:
                    break
        
        # Deteksi impersonation (pengirim mengaku sebagai perusahaan lain)
        impersonation_keywords = [
            'security team', 'support team', 'customer service', 'billing department',
            'account department', 'verification team', 'fraud department'
        ]
        features['sender_impersonation'] = 0
        for keyword in impersonation_keywords:
            if keyword in text_content.lower() and sender_domain not in legitimate_domains:
                features['sender_impersonation'] = 1
                break
    else:
        features['sender_content_mismatch'] = 0
        features['sender_impersonation'] = 0

    return features

def extract_email_security_features(text):
    features = {}

    # Deteksi klaim keamanan berlebihan
    security_claims = [
        '100% secure', 'completely safe', 'guaranteed secure',
        'bank-level security', 'military-grade encryption',
        'end-to-end encrypted', 'ssl secured', 'https secured'
    ]
    features['excessive_security_claims'] = sum(1 for claim in security_claims if claim in text.lower())

    # Deteksi permintaan verifikasi yang mencurigakan
    verification_requests = [
        'verify your account', 'verify your identity', 'verify now',
        'confirm your account', 'confirm your identity', 'confirm now',
        'validate your account', 'validate your identity'
    ]
    features['suspicious_verification_request'] = sum(1 for request in verification_requests if request in text.lower())

    # Deteksi permintaan informasi sensitif
    sensitive_info_requests = [
        'provide your password', 'enter your pin', 'input your cvv',
        'send your card number', 'share your ssn', 'disclose your account details'
    ]
    features['sensitive_info_request'] = sum(1 for request in sensitive_info_requests if request in text.lower())

    # Deteksi ancaman akun
    account_threats = [
        'account will be suspended', 'account will be closed',
        'account will be terminated', 'account will be blocked',
        'your account is at risk', 'your account has been compromised'
    ]
    features['account_threat_count'] = sum(1 for threat in account_threats if threat in text.lower())

    return features

# <--- PERBAIKAN 2: GANTI SELURUH FUNGSI generate_explanation ---
def generate_explanation(features, prediction_status, prob_phishing, prob_safe):
    """
    Generate explanation for the prediction based on features and prediction status.
    prediction_status can be: "phishing", "safe", or "suspicious"
    """
    explanation = []
    
    if prediction_status == "phishing":
        # --- ZONA MERAH: PASTI PHISHING ---
        explanation.append(f"🚨 Email ini terdeteksi sebagai **Phishing** dengan tingkat kepercayaan {prob_phishing*100:.0f}%.")
        explanation.append("\n**Indikator Bahaya yang Terdeteksi:**")
        
        indicators_found = []
        
        # Cek setiap fitur dan tambahkan ke daftar jika terdeteksi
        if features.get('suspicious_keyword_count', 0) > 0:
            indicators_found.append(f"Mengandung {features['suspicious_keyword_count']} kata kunci mencurigakan (seperti 'urgent', 'verify account', 'suspended')")
        
        if features.get('personal_info_request', 0) == 1:
            indicators_found.append("Meminta informasi pribadi sensitif (password, PIN, nomor kartu kredit)")
        
        if features.get('has_threat', 0) == 1:
            indicators_found.append("Menggunakan bahasa yang mengancam (suspend, terminate, close account)")
        
        if features.get('has_suspicious_attachment', 0) == 1:
            indicators_found.append("Mengandung lampiran dengan ekstensi mencurigakan (.exe, .zip, .scr)")
        
        if features.get('has_suspicious_short_domain', 0) == 1 or features.get('has_typosquatting', 0) == 1:
            indicators_found.append("Menggunakan domain mencurigakan atau mirip dengan domain resmi")
        
        if features.get('urgency_word_count', 0) > 2:
            indicators_found.append(f"Menciptakan rasa urgensi berlebihan ({features['urgency_word_count']} kata mendesak)")
        
        if features.get('excessive_exclamation', 0) == 1:
            indicators_found.append(f"Penggunaan tanda seru berlebihan ({features.get('exclamation_count', 0)} kali)")
        
        if features.get('has_misleading_link', 0) == 1:
            indicators_found.append("Mengandung link dengan teks anchor yang menyesatkan")
        
        if features.get('sender_impersonation', 0) == 1:
            indicators_found.append("Pengirim mencoba menyamar sebagai entitas resmi")
        
        if features.get('has_ip_url', 0) == 1:
            indicators_found.append("Menggunakan alamat IP sebagai URL (bukan domain)")
        
        if features.get('sensitive_info_request', 0) > 0:
            indicators_found.append("Meminta Anda mengirimkan data sensitif melalui email")
        
        if features.get('account_threat_count', 0) > 0:
            indicators_found.append("Mengancam akan menutup/memblokir akun Anda")
        
        # Tambahkan indikator ke explanation
        if indicators_found:
            for indicator in indicators_found:
                explanation.append(f"• {indicator}")
        else:
            # Jika tidak ada indikator spesifik, berikan penjelasan umum
            explanation.append("• Pola keseluruhan email ini sangat mirip dengan modus phishing yang telah dikenal")
            explanation.append("• Kombinasi struktur, kata-kata, dan metadata menunjukkan karakteristik phishing")
        
        explanation.append("\n⚠️ **Rekomendasi:** JANGAN klik link apapun, jangan berikan informasi pribadi, dan hapus email ini segera.")

    elif prediction_status == "safe":
        # --- ZONA HIJAU: PASTI AMAN ---
        explanation.append(f"✅ Email ini terdeteksi sebagai **Aman** dengan tingkat kepercayaan {prob_safe*100:.0f}%.")
        explanation.append("\n**Indikator Keamanan:**")
        
        safety_indicators = []
        
        if features.get('is_legitimate_domain', 0) == 1:
            safety_indicators.append(f"Pengirim menggunakan domain resmi yang terpercaya ({features.get('sender_domain', 'N/A')})")
        
        if features.get('suspicious_keyword_count', 0) == 0:
            safety_indicators.append("Tidak mengandung kata kunci phishing yang mencurigakan")
        
        if features.get('personal_info_request', 0) == 0:
            safety_indicators.append("Tidak meminta informasi pribadi sensitif")
        
        if features.get('has_threat', 0) == 0:
            safety_indicators.append("Tidak menggunakan bahasa yang mengancam")
        
        if features.get('has_suspicious_attachment', 0) == 0:
            safety_indicators.append("Tidak mengandung lampiran mencurigakan")
        
        if features.get('urgency_word_count', 0) == 0:
            safety_indicators.append("Tidak ada tekanan waktu atau urgensi yang berlebihan")
        
        if features.get('sender_impersonation', 0) == 0:
            safety_indicators.append("Tidak ada tanda-tanda impersonasi")
        
        # Tambahkan indikator ke explanation
        if safety_indicators:
            for indicator in safety_indicators:
                explanation.append(f"• {indicator}")
        else:
            explanation.append("• Email tidak menunjukkan karakteristik phishing yang umum")
            explanation.append("• Struktur dan konten email terlihat normal dan legitim")
        
        explanation.append("\n✓ **Rekomendasi:** Email ini tampak aman, namun tetap waspada terhadap link yang tidak dikenal.")

    else:  # "suspicious"
        # --- ZONA ABU-ABU: MENCURIGAKAN ---
        explanation.append(f"⚡ Email ini tergolong **Mencurigakan** dengan tingkat kecurigaan {prob_phishing*100:.0f}%.")
        explanation.append("\n**Faktor yang Membuat Email Ini Mencurigakan:**")
        
        suspicious_factors = []
        
        # Analisis fitur yang membuat email mencurigakan
        if features.get('capital_word_ratio', 0) > 0.2:
            suspicious_factors.append(f"Penggunaan huruf kapital tidak wajar ({features['capital_word_ratio']*100:.0f}% kata dalam KAPITAL)")
        
        if features.get('exclamation_count', 0) > 3:
            suspicious_factors.append(f"Terlalu banyak tanda seru ({features['exclamation_count']} kali)")
        
        if features.get('urgency_word_count', 0) > 0:
            suspicious_factors.append(f"Mengandung kata-kata yang mendesak atau bersifat urgensi ({features['urgency_word_count']} kata)")
        
        if features.get('fear_intensity', 0) > 0:
            suspicious_factors.append(f"Menggunakan kata-kata yang memicu rasa takut ({features['fear_intensity']} kata)")
        
        if features.get('greed_trigger', 0) > 0:
            suspicious_factors.append(f"Menggunakan kata-kata yang memicu keserakahan ({features['greed_trigger']} kata seperti 'free', 'win', 'prize')")
        
        if features.get('url_count', 0) > 2:
            suspicious_factors.append(f"Mengandung banyak link ({features['url_count']} link) yang perlu diverifikasi")
        
        if features.get('has_generic_greeting', 0) == 1:
            suspicious_factors.append("Menggunakan salam generik ('Dear Customer') bukan nama personal")
        
        if features.get('suspicious_keyword_count', 0) > 0:
            suspicious_factors.append(f"Mengandung {features['suspicious_keyword_count']} kata kunci yang perlu diwaspadai")
        
        if features.get('has_time_limit', 0) == 1:
            suspicious_factors.append("Memberikan batasan waktu yang ketat untuk bertindak")
        
        if features.get('action_request_count', 0) > 3:
            suspicious_factors.append(f"Terlalu banyak permintaan tindakan ({features['action_request_count']} kali)")
        
        if features.get('is_free_email', 0) == 1 and features.get('brand_mention_count', 0) > 0:
            suspicious_factors.append("Pengirim menggunakan email gratis namun mengaku dari perusahaan besar")
        
        if features.get('sender_content_mismatch', 0) == 1:
            suspicious_factors.append("Ada ketidaksesuaian antara pengirim dan konten email")
        
        # Tambahkan faktor ke explanation
        if suspicious_factors:
            for factor in suspicious_factors:
                explanation.append(f"• {factor}")
        else:
            # Jika tidak ada faktor spesifik yang jelas
            explanation.append("• Meskipun tidak mengandung elemen phishing yang jelas, pola keseluruhan email ini menunjukkan karakteristik yang tidak biasa")
            explanation.append("• Kombinasi beberapa faktor kecil menciptakan profil mencurigakan secara keseluruhan")
            explanation.append(f"• Sistem mendeteksi {prob_phishing*100:.0f}% kesamaan dengan pola phishing, namun tidak cukup tinggi untuk dikategorikan sebagai phishing pasti")
        
        explanation.append("\n⚠️ **Rekomendasi:** Berhati-hati! Verifikasi pengirim sebelum mengklik link atau memberikan informasi. Jika ragu, hubungi perusahaan langsung melalui saluran resmi.")
    
    return explanation


@app.route('/')
def index():
    return render_template('./templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get email content from request
        data = request.json
        email_content = data.get('email_content', '')
        
        if not email_content:
            return jsonify({'error': 'Email content is required'}), 400
        
        # Preprocess the email
        cleaned_text, extracted_date, extracted_sender = enhanced_preprocess_combined_text(email_content)
        
        # Extract features
        phishing_features = extract_phishing_features(email_content)
        url_features = extract_url_features(email_content)
        brand_features = extract_brand_features(email_content)
        sender_features = extract_sender_features(extracted_sender)
        extension_features = extract_file_extension_features(email_content)
        advanced_sender = advanced_sender_analysis(extracted_sender, email_content)
        security_features = extract_email_security_features(email_content)
        
        # Combine all features
        all_features = {**phishing_features, **url_features, **brand_features, 
                       **sender_features, **extension_features, **advanced_sender, 
                       **security_features}
        
        # Add text length feature
        all_features['text_length'] = len(cleaned_text.split())
        all_features['has_attachment'] = 1 if 'attached file' in email_content.lower() else 0
        
        # Handle date features
        try:
            parsed_date = pd.to_datetime(extracted_date, format='%a %b %d %Y', errors='coerce')
            all_features['is_weekend'] = int(parsed_date.dayofweek >= 5) if not pd.isna(parsed_date) else 0
            all_features['hour_sent'] = int(parsed_date.hour) if not pd.isna(parsed_date) else 12
        except:
            all_features['is_weekend'] = 0
            all_features['hour_sent'] = 12
        
        # Create dataframe with the features
        df_features = pd.DataFrame([all_features])
        
        # Select only the numeric features used in training
        X_numeric = df_features[numeric_features].fillna(0)
        
        # Convert to sparse matrix
        X_numeric_sparse = csr_matrix(X_numeric.values)
        
        # Transform text using TF-IDF
        X_tfidf = tfidf.transform([cleaned_text])
        
        # Combine TF-IDF and numeric features
        X_combined = hstack([X_tfidf, X_numeric_sparse])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X_combined)[0]
        prob_safe = float(probabilities[0])
        prob_phishing = float(probabilities[1])

        # Define thresholds for classification
        PHISHING_THRESHOLD_HIGH = 0.75  # ≥75% = Phishing
        SAFE_THRESHOLD = 0.40           # <40% = Safe
        # 40%-74% = Suspicious

        # Determine prediction status
        if prob_phishing >= PHISHING_THRESHOLD_HIGH:
            prediction_status = "phishing"
        elif prob_phishing < SAFE_THRESHOLD:
            prediction_status = "safe"
        else:
            prediction_status = "suspicious"

        # Generate explanation
        explanation = generate_explanation(
            features=all_features,
            prediction_status=prediction_status,
            prob_phishing=prob_phishing,
            prob_safe=prob_safe
        )
        
        # Return results
        result = {
            'prediction_status': prediction_status,
            'phishing_probability': round(prob_phishing, 4),
            'safe_probability': round(prob_safe, 4),
            'explanation': explanation,
            'extracted_sender': extracted_sender,
            'extracted_date': extracted_date,
            'thresholds': {
                'phishing_threshold': PHISHING_THRESHOLD_HIGH,
                'safe_threshold': SAFE_THRESHOLD
            },
            'feature_summary': {
                'suspicious_keywords': all_features.get('suspicious_keyword_count', 0),
                'urgency_words': all_features.get('urgency_word_count', 0),
                'urls': all_features.get('url_count', 0),
                'exclamations': all_features.get('exclamation_count', 0)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Terjadi kesalahan saat memproses email',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_type': model_metadata['model_type'],
        'version': model_metadata['version'],
        'creation_date': model_metadata['creation_date']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)