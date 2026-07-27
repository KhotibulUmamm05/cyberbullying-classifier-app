import re

import joblib
import numpy as np
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

st.set_page_config(page_title="Cyberbullying Tweet Classifier", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/best_model_rf.pkl")
    tfidf = joblib.load("model/tfidf_vectorizer.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    numerical_features = joblib.load("model/numerical_features.pkl")
    return model, tfidf, scaler, encoder, numerical_features


stop_words, lemmatizer = load_nltk_resources()
model, tfidf, scaler, encoder, numerical_features = load_artifacts()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def extract_numerical_features(raw_text, clean):
    values = {
        "tweet_length": len(clean),
        "word_count": len(clean.split()),
        "capital_letters": sum(1 for c in raw_text if c.isupper()),
        "exclamation_count": raw_text.count("!"),
        "question_count": raw_text.count("?"),
        "mention_count": raw_text.count("@"),
        "hashtag_count": raw_text.count("#"),
    }
    return np.array([[values[col] for col in numerical_features]])


st.title("🛡️ Cyberbullying Tweet Classifier")
st.write(
    "Masukkan teks tweet untuk memprediksi apakah teks tersebut termasuk "
    "kategori cyberbullying (age, ethnicity, gender, religion, other_cyberbullying) "
    "atau not_cyberbullying."
)

user_input = st.text_area("Teks Tweet", height=140, placeholder="Tulis atau tempel teks tweet di sini...")

if st.button("Prediksi", type="primary"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks tweet terlebih dahulu.")
    else:
        clean = clean_text(user_input)

        if clean.strip() == "":
            st.error(
                "Teks tidak mengandung kata yang valid untuk dianalisis "
                "setelah proses pembersihan (kemungkinan hanya berisi URL/mention/simbol)."
            )
        else:
            num_vector = extract_numerical_features(user_input, clean)

            text_vector = tfidf.transform([clean])
            num_scaled = scaler.transform(num_vector)
            X_input = hstack([text_vector, num_scaled])

            pred_idx = model.predict(X_input)[0]
            pred_label = encoder.inverse_transform([pred_idx])[0]

            st.subheader("Hasil Prediksi")
            st.success(f"Kategori: **{pred_label}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                proba_dict = {encoder.classes_[i]: float(proba[i]) for i in range(len(encoder.classes_))}
                st.write("Probabilitas per kategori:")
                st.bar_chart(proba_dict)

            with st.expander("Lihat teks setelah dibersihkan (clean_text)"):
                st.write(clean)
