import json
import os
import re

import joblib
import numpy as np
import pandas as pd
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


@st.cache_data
def load_dashboard_data():
    label_counts = pd.read_csv("dashboard_data/label_counts.csv", index_col=0)["count"]
    tweet_lengths = pd.read_csv("dashboard_data/tweet_lengths.csv")["tweet_length"]
    with open("dashboard_data/summary_stats.json") as f:
        summary_stats = json.load(f)
    return label_counts, tweet_lengths, summary_stats


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

tab_predict, tab_eda = st.tabs(["🔍 Prediksi", "📊 Analisis Dataset"])

with tab_predict:
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

with tab_eda:
    st.write(
        "Ringkasan hasil analisis dataset **Cyberbullying Tweets** "
        "(setelah dibersihkan dari duplikat & data kosong) yang digunakan untuk melatih model."
    )

    if not os.path.exists("dashboard_data/summary_stats.json"):
        st.warning(
            "Data dashboard belum tersedia. Jalankan cell export dashboard di notebook "
            "(bagian setelah Data Cleaning) untuk menghasilkan folder `dashboard_data/`."
        )
    else:
        label_counts, tweet_lengths, summary_stats = load_dashboard_data()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", f"{summary_stats['total_data']:,}")
        col2.metric("Jumlah Kategori", summary_stats["jumlah_kategori"])
        col3.metric("Rata-rata Panjang Tweet", f"{summary_stats['rata_rata_panjang_tweet']:.0f} karakter")

        st.subheader("Distribusi Label")
        st.bar_chart(label_counts)

        st.subheader("Distribusi Panjang Tweet")
        bins = pd.cut(
            tweet_lengths[tweet_lengths <= 300],
            bins=range(0, 320, 20),
            right=False,
        )
        hist_counts = bins.value_counts().sort_index()
        hist_counts.index = [int(interval.left) for interval in hist_counts.index]
        hist_counts.index.name = "Panjang tweet (karakter)"
        st.bar_chart(hist_counts)
        st.caption(
            f"Ditampilkan untuk tweet dengan panjang ≤ 300 karakter. "
            f"Tweet terpanjang: {summary_stats['tweet_terpanjang']:,} karakter, "
            f"tweet terpendek: {summary_stats['tweet_terpendek']} karakter."
        )

        st.subheader("Word Cloud")
        if os.path.exists("dashboard_data/wordcloud.png"):
            st.image("dashboard_data/wordcloud.png", width="stretch")
