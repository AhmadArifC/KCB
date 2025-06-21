import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

# Memuat model yang telah disimpan
model = load_model('spam_email_detection_model.keras')

# Load the training data to get feature columns
train_data = pd.read_excel('/content/drive/MyDrive/colab/spambase_train.xlsx')
X_train = train_data.drop('class', axis=1)

# Scaler yang sama yang digunakan saat pelatihan
scaler = StandardScaler()
scaler.fit(X_train)

# Fungsi untuk memproses input email dan memprediksi spam atau tidak
def predict_email_spam(email_text):
    email_features_dict = {col: 0.0 for col in X_train.columns}

    email_words = re.findall(r'\b\w+\b', email_text.lower())
    total_words = len(email_words)
    word_counts = {}
    for word in email_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    for word_freq_col in [col for col in X_train.columns if col.startswith('word_freq_')]:
        word = word_freq_col.replace('word_freq_', '')
        if word in word_counts:
            email_features_dict[word_freq_col] = (word_counts[word] / total_words) * 100 if total_words > 0 else 0

    total_chars = len(email_text)
    char_counts = {}
    for char in email_text:
        char_counts[char] = char_counts.get(char, 0) + 1

    for char_freq_col in [col for col in X_train.columns if col.startswith('char_freq_')]:
        char = char_freq_col.replace('char_freq_', '')
        if char in char_counts:
            email_features_dict[char_freq_col] = (char_counts[char] / total_chars) * 100 if total_chars > 0 else 0

    capital_runs = re.findall(r'[A-Z]+', email_text)
    capital_run_lengths = [len(run) for run in capital_runs]

    if capital_run_lengths:
        email_features_dict['capital_run_length_average'] = np.mean(capital_run_lengths)
        email_features_dict['capital_run_length_longest'] = np.max(capital_run_lengths)
        email_features_dict['capital_run_length_total'] = np.sum(capital_run_lengths)
    else:
        email_features_dict['capital_run_length_average'] = 0.0
        email_features_dict['capital_run_length_longest'] = 0.0
        email_features_dict['capital_run_length_total'] = 0.0

    email_features = np.array([email_features_dict[col] for col in X_train.columns]).reshape(1, -1)
    email_features_scaled = scaler.transform(email_features)
    email_features_scaled = np.expand_dims(email_features_scaled, axis=-1)

    prediction = model.predict(email_features_scaled)

    if prediction > 0.5:
        return "Spam"
    else:
        return "Not Spam"

# Streamlit UI
st.title("Email Spam Detection")
st.write("Enter the email content below to check if it's spam or not:")

email_input = st.text_area("Email Content", height=200)
if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter an email content to predict!")
    else:
        result = predict_email_spam(email_input)
        st.success(f"Prediction: {result}")
