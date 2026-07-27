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

st.set_page_config(
    page_title="Cyberbullying Tweet Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
[data-testid="stSidebar"] .stButton > button {
    border-radius: 999px;
    width: 100%;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.hero {
    background: linear-gradient(135deg, #5B5FEF 0%, #4347C4 100%);
    padding: 2.2rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { color: white; margin-bottom: 0.4rem; font-size: 2.1rem; }
.hero p { color: #E7E8FD; font-size: 1.05rem; margin: 0; }

.section-card {
    background: #F2F3FB;
    border-left: 6px solid #5B5FEF;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.2rem;
}
.section-card h4 { margin-top: 0; color: #2B2F77; }
.section-card p { color: #33364D; margin-bottom: 0; }

.stat-card {
    background: #FFFFFF;
    border: 1px solid #E4E5F5;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(91, 95, 239, 0.08);
}
.stat-card .stat-value { font-size: 1.8rem; font-weight: 700; color: #5B5FEF; }
.stat-card .stat-label { font-size: 0.9rem; color: #6B7280; margin-top: 0.2rem; }

.result-safe {
    background: #DCFCE7;
    border-left: 6px solid #16A34A;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.result-safe h3 { color: #15803D; margin: 0 0 0.3rem 0; }
.result-safe p { color: #166534; margin: 0; }

.result-flag {
    background: #FEE2E2;
    border-left: 6px solid #DC2626;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.result-flag h3 { color: #B91C1C; margin: 0 0 0.3rem 0; }
.result-flag p { color: #991B1B; margin: 0; }

.step-item {
    background: #FFFFFF;
    border: 1px solid #E4E5F5;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}

.note-box {
    background: #FEF9C3;
    border-left: 6px solid #CA8A04;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}
.note-box p { color: #713F12; margin: 0; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    sample_data = pd.read_csv("dashboard_data/sample_data.csv")
    with open("dashboard_data/summary_stats.json") as f:
        summary_stats = json.load(f)
    return label_counts, tweet_lengths, sample_data, summary_stats


stop_words, lemmatizer = load_nltk_resources()
model, tfidf, scaler, encoder, numerical_features = load_artifacts()

DASHBOARD_AVAILABLE = os.path.exists("dashboard_data/summary_stats.json")
if DASHBOARD_AVAILABLE:
    label_counts, tweet_lengths, sample_data, summary_stats = load_dashboard_data()


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


# ---------- Sidebar Navigation (pill-style buttons) ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"

NAV_PAGES = ["Home", "Analisis Dataset", "Prediction"]

with st.sidebar:
    st.markdown("## Cyberbullying Classifier")
    current_page = st.session_state.page
    for nav_page in NAV_PAGES:
        if st.button(
            nav_page,
            key=f"nav_{nav_page}",
            type="primary" if current_page == nav_page else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = nav_page
            st.rerun()
    st.markdown("---")
    st.caption("Final Project — Klasifikasi Cyberbullying pada Tweet\n\nModel: Random Forest")

page = st.session_state.page


# ---------- Home Page ----------
def render_home():
    st.markdown(
        """
        <div class="hero">
            <h1>Cyberbullying Tweet Classifier</h1>
            <p>Aplikasi klasifikasi teks tweet untuk mendeteksi kategori cyberbullying menggunakan Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <h4>Tentang Proyek</h4>
            <p>Proyek ini membangun model klasifikasi teks untuk mengidentifikasi apakah sebuah tweet mengandung
            unsur <b>cyberbullying</b> beserta kategorinya (age, ethnicity, gender, religion, other_cyberbullying)
            atau termasuk <b>not_cyberbullying</b>. Model dilatih menggunakan dataset publik
            <i>Cyberbullying Tweets</i> yang telah melalui proses pembersihan data (penghapusan duplikat, data
            kosong, dan normalisasi teks) sebelum divektorisasi dengan TF-IDF dan dilatih menggunakan beberapa
            algoritma Machine Learning.</p>
        </div>
        <div class="section-card">
            <h4>Tujuan Aplikasi</h4>
            <p>Aplikasi ini dibuat untuk membantu <b>mendeteksi dan mengklasifikasikan konten cyberbullying secara
            otomatis</b> pada teks tweet, sehingga dapat digunakan sebagai alat bantu moderasi konten, penelitian,
            maupun edukasi mengenai penyebaran ujaran kebencian di media sosial.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Ringkasan Model & Data")
    if DASHBOARD_AVAILABLE:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{summary_stats["total_data"]:,}</div>'
                f'<div class="stat-label">Total Data Tweet</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{summary_stats["jumlah_kategori"]}</div>'
                f'<div class="stat-label">Jumlah Kategori</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="stat-card"><div class="stat-value">Random Forest</div>'
                '<div class="stat-label">Model yang Digunakan</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("Data ringkasan belum tersedia. Jalankan cell ekspor dashboard di notebook terlebih dahulu.")

    st.markdown("#### Cara Menggunakan Aplikasi")
    steps = [
        "Buka halaman <b>Analisis Dataset</b> untuk memahami karakteristik data yang digunakan melatih model.",
        "Buka halaman <b>Prediction</b>, lalu masukkan teks tweet yang ingin diperiksa.",
        "Klik tombol <b>Prediksi</b> untuk melihat kategori cyberbullying beserta probabilitasnya.",
    ]
    for i, step in enumerate(steps, start=1):
        st.markdown(f'<div class="step-item"><b>{i}.</b> {step}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note-box">
            <p><b>Catatan:</b> Gunakan teks berbahasa Inggris saat mencoba fitur prediksi, karena model
            dilatih menggunakan data tweet berbahasa Inggris sehingga hasil prediksi untuk teks berbahasa
            lain kemungkinan besar tidak akurat.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Analisis Dataset Page ----------
def render_dataset_analysis():
    st.markdown("## Analisis Dataset")
    st.write(
        "Bagian ini menjelaskan pemahaman terhadap dataset **Cyberbullying Tweets** yang digunakan untuk "
        "melatih model, setelah melalui proses pembersihan (penghapusan duplikat & data kosong)."
    )

    if not DASHBOARD_AVAILABLE:
        st.warning(
            "Data dashboard belum tersedia. Jalankan cell export dashboard di notebook "
            "(bagian setelah Data Cleaning) untuk menghasilkan folder `dashboard_data/`."
        )
        return

    st.markdown("### Data Understanding")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{summary_stats['total_data']:,}")
    c2.metric("Jumlah Kategori", summary_stats["jumlah_kategori"])
    c3.metric("Rata-rata Panjang Tweet", f"{summary_stats['rata_rata_panjang_tweet']:.0f} karakter")

    st.markdown("**Contoh data tiap kategori:**")
    st.dataframe(sample_data, width="stretch", hide_index=True)

    st.markdown("**Statistik ringkas panjang tweet (karakter):**")
    length_stats = pd.DataFrame(
        {
            "Statistik": ["Tweet Terpanjang", "Tweet Terpendek", "Rata-rata"],
            "Nilai (karakter)": [
                summary_stats["tweet_terpanjang"],
                summary_stats["tweet_terpendek"],
                round(summary_stats["rata_rata_panjang_tweet"], 1),
            ],
        }
    )
    st.dataframe(length_stats, width="stretch", hide_index=True)

    st.markdown("### Distribusi Label")
    st.dataframe(label_counts.rename("Jumlah").to_frame(), width="stretch")
    st.bar_chart(label_counts)

    st.markdown("### Distribusi Panjang Tweet")
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

    st.markdown("### Word Cloud")
    if os.path.exists("dashboard_data/wordcloud.png"):
        st.image("dashboard_data/wordcloud.png", width="stretch")


# ---------- Prediction Page ----------
def render_prediction():
    st.markdown("## Prediction")
    st.write(
        "Masukkan teks tweet untuk memprediksi apakah teks tersebut termasuk kategori cyberbullying "
        "(age, ethnicity, gender, religion, other_cyberbullying) atau not_cyberbullying."
    )
    st.markdown(
        """
        <div class="note-box">
            <p><b>Catatan:</b> Gunakan teks berbahasa Inggris, karena model dilatih menggunakan data tweet
            berbahasa Inggris sehingga hasil prediksi untuk teks berbahasa lain kemungkinan besar tidak akurat.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_input = st.text_area("Teks Tweet", height=140, placeholder="Write or paste tweet text in English here...")

    if not st.button("Prediksi", type="primary"):
        return

    if not user_input.strip():
        st.warning("Silakan masukkan teks tweet terlebih dahulu.")
        return

    clean = clean_text(user_input)

    if clean.strip() == "":
        st.error(
            "Teks tidak mengandung kata yang valid untuk dianalisis "
            "setelah proses pembersihan (kemungkinan hanya berisi URL/mention/simbol)."
        )
        return

    num_vector = extract_numerical_features(user_input, clean)
    text_vector = tfidf.transform([clean])
    num_scaled = scaler.transform(num_vector)
    X_input = hstack([text_vector, num_scaled])

    pred_idx = model.predict(X_input)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    proba = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
    top_proba = float(proba[pred_idx]) if proba is not None else None

    st.markdown("### Hasil Prediksi")
    card_class = "result-safe" if pred_label == "not_cyberbullying" else "result-flag"
    proba_text = f" ({top_proba * 100:.1f}%)" if top_proba is not None else ""
    st.markdown(
        f'<div class="{card_class}"><h3>{pred_label}{proba_text}</h3>'
        f'<p>Kategori dengan probabilitas tertinggi hasil prediksi model.</p></div>',
        unsafe_allow_html=True,
    )

    if proba is not None:
        st.markdown("#### Probabilitas per Kategori")
        proba_df = (
            pd.DataFrame(
                {
                    "Kategori": encoder.classes_,
                    "Probabilitas (%)": proba * 100,
                }
            )
            .sort_values("Probabilitas (%)", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            proba_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Probabilitas (%)": st.column_config.ProgressColumn(
                    "Probabilitas (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                )
            },
        )

    with st.expander("Lihat teks setelah dibersihkan (clean_text)"):
        st.write(clean)


# ---------- Router ----------
if page == "Home":
    render_home()
elif page == "Analisis Dataset":
    render_dataset_analysis()
else:
    render_prediction()
