# app.py
# Deployment Sistem Deteksi Kemiripan Judul Skripsi

import subprocess
import sys

try:
    import PySastrawi
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PySastrawi", "-q"])

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Deteksi Kemiripan Judul Skripsi",
    layout="wide",
)

# LOAD CSS EKSTERNAL
with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# SESSION STATE — riwayat pencarian
if "history" not in st.session_state:
    st.session_state.history = []


# LOAD SASTRAWI
@st.cache_resource
def load_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer          = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover


# LOAD MODEL
@st.cache_resource
def load_model():
    loaded        = joblib.load("models/tfidf_model.joblib")
    vectorizer    = loaded['vectorizer']
    tfidf_matrix  = loaded['tfidf_matrix']
    docs          = loaded['docs']
    df            = pd.read_csv("data/judul_cleansing.csv")
    docs_original = df['judul_after_clean'].tolist()
    return vectorizer, tfidf_matrix, docs, docs_original


# HELPERS
def kategorisasi(score: float) -> tuple:
    if score >= 0.55:
        return "High Similarity", "badge-high"
    elif score >= 0.30:
        return "Medium",          "badge-medium"
    elif score >= 0.10:
        return "Low Similarity",  "badge-low"
    else:
        return "No Similarity",   "badge-no"


def bar_color(badge: str) -> str:
    return {
        "badge-high":   "#34d399",
        "badge-medium": "#fbbf24",
        "badge-low":    "#f87171",
        "badge-no":     "#334155",
    }.get(badge, "#334155")


def highlight_keywords(title: str, keywords: list) -> str:
    """Bungkus kata kunci (hasil preprocessing) dalam tag <mark>."""
    import re
    result = title
    for kw in sorted(keywords, key=len, reverse=True):
        if len(kw) < 3:
            continue
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", result)
    return result


def detect_similarity(query, vectorizer, tfidf_matrix, docs, docs_original,
                      stemmer, stopword_remover, top_n=10, threshold=0.0):
    query_clean  = preprocess(query, stemmer, stopword_remover)
    query_vector = vectorizer.transform([query_clean])
    scores       = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_idx      = np.argsort(scores)[::-1][:top_n]

    results = []
    for idx in top_idx:
        s = float(scores[idx])
        if s < threshold:
            continue
        label, badge = kategorisasi(s)
        results.append({
            "judul": docs_original[idx],
            "skor":  round(s, 4),
            "label": label,
            "badge": badge,
        })
    return query_clean, results


def results_to_excel(results: list) -> bytes:
    df_out = pd.DataFrame([
        {"Rank": i+1, "Judul": r["judul"], "Skor": r["skor"], "Kategori": r["label"]}
        for i, r in enumerate(results)
    ])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Hasil")
    return buf.getvalue()


# LOAD RESOURCES
try:
    stemmer, stopword_remover = load_sastrawi()
    vectorizer, tfidf_matrix, docs, docs_original = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan file `models/tfidf_model.joblib` tersedia.")
    st.stop()


# HERO HEADER
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-orb hero-orb-1"></div>
  <div class="hero-orb hero-orb-2"></div>
  <div class="hero-orb hero-orb-3"></div>
  <div class="hero-content">
    <div class="hero-tag">
      <span class="hero-tag-dot"></span>
      TF-IDF · Cosine Similarity · NLP
    </div>
    <h1 class="hero-title">Sistem Deteksi Kemiripan<br>Judul Skripsi</h1>
    <p class="hero-sub">
      Masukkan judul skripsi Anda untuk mendeteksi tingkat kemiripan
      dengan {len(docs_original)} judul yang telah terdaftar dalam database.
    </p>
    <div class="hero-stats">
      <div class="hstat">
        <div class="hstat-val">{len(docs_original)}</div>
        <div class="hstat-lbl">Judul Database</div>
      </div>
      <div class="hstat">
        <div class="hstat-val">6</div>
        <div class="hstat-lbl">Tahap Proses</div>
      </div>
      <div class="hstat">
        <div class="hstat-val">4</div>
        <div class="hstat-lbl">Level Similarity</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# PIPELINE STRIP
steps = ["Cleansing", "Case Folding", "Stopword Removal",
         "Stemming", "TF-IDF Vectorization", "Cosine Similarity"]
chips = "".join(
    f'<div class="pipe-step"><span class="pipe-num">{i+1}</span>{s}</div>'
    for i, s in enumerate(steps)
)
st.markdown(f'<div class="pipeline-wrap">{chips}</div>', unsafe_allow_html=True)


# RIWAYAT PENCARIAN
if st.session_state.history:
    st.markdown(
        '<div class="sec-heading">Riwayat Pencarian '
        '<span class="sec-badge-new">Terbaru</span></div>',
        unsafe_allow_html=True,
    )
    chips_html = ""
    for h in st.session_state.history[-5:][::-1]:
        short = h[:]
        chips_html += f'<span class="hist-chip"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="flex-shrink:0"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> {short}</span>'
    st.markdown(f'<div class="history-wrap">{chips_html}</div>', unsafe_allow_html=True)

# SEARCH CARD DAN INPUT
st.markdown('<div class="search-label">Masukkan Judul Skripsi</div>', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        label="Judul Skripsi",
        placeholder="Contoh: Sistem Informasi Penjualan Berbasis Web Menggunakan Framework Laravel",
        label_visibility="collapsed",
    )
with col2:
    cari = st.button("Cari")


# SIDEBAR
with st.sidebar:
    st.markdown("### Pengaturan")

    top_n = st.select_slider(
        "Tampilkan top-N hasil",
        options=[5, 10, 20, 50],
        value=10,
    )

    threshold = st.slider(
        "Minimum skor kemiripan",
        min_value=0.00,
        max_value=1.00,
        value=0.00,
        step=0.01,
        format="%.2f",
        help="Hasil dengan skor di bawah nilai ini akan disembunyikan.",
    )
    st.caption("Geser slider untuk menyembunyikan hasil kemiripan rendah.")

    st.markdown("---")
    st.markdown("### Kategori Skor")
    st.markdown("""
<div>
  <div class="score-row"><span><span class="dot d-green"></span>High Similarity</span><span>≥ 0.55</span></div>
  <div class="score-row"><span><span class="dot d-amber"></span>Medium</span><span>0.30 – 0.55</span></div>
  <div class="score-row"><span><span class="dot d-red"></span>Low Similarity</span><span>0.10 – 0.30</span></div>
  <div class="score-row"><span><span class="dot d-gray"></span>No Similarity</span><span>&lt; 0.10</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Tentang Model")
    st.markdown(f"""
- **Model:** TF-IDF + Cosine Similarity
- **Database:** {len(docs_original)} judul skripsi
- **Metrik:** MAE, RMSE, Spearman Rank Correlation
""")

# PROSES & TAMPILKAN HASIL
if cari and query.strip():

    if not st.session_state.history or st.session_state.history[-1] != query.strip():
        st.session_state.history.append(query.strip())

    with st.spinner("Memproses..."):
        query_clean, results = detect_similarity(
            query, vectorizer, tfidf_matrix, docs, docs_original,
            stemmer, stopword_remover,
            top_n=top_n,
            threshold=threshold,
        )

    #  Preview preprocessing 
    st.markdown(f"""
<div class="preview-box">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;margin-top:2px"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg>
  <div><strong>Hasil Preprocessing:</strong><br>{query_clean}</div>
</div>
""", unsafe_allow_html=True)

    if not results:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon-wrap">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
  </div>
  <div class="empty-title">Tidak ada hasil yang memenuhi threshold</div>
  <div class="empty-sub">
    Coba turunkan nilai minimum skor di sidebar,
    atau periksa kembali judul yang dimasukkan.
  </div>
</div>
""", unsafe_allow_html=True)

    else:
        #  Metric cards 
        counts = {"badge-high": 0, "badge-medium": 0, "badge-low": 0, "badge-no": 0}
        for r in results:
            counts[r["badge"]] += 1

        st.markdown(
            '<div class="sec-heading">Ringkasan Hasil '
            '<span class="sec-badge-new">Baru</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
<div class="metrics-row">
  <div class="metric-card mc-green">
    <span class="metric-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="1.8"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></span>
    <div class="metric-val mv-green">{counts['badge-high']}</div>
    <div class="metric-label">High Similarity</div>
  </div>
  <div class="metric-card mc-amber">
    <span class="metric-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></span>
    <div class="metric-val mv-amber">{counts['badge-medium']}</div>
    <div class="metric-label">Medium</div>
  </div>
  <div class="metric-card mc-red">
    <span class="metric-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg></span>
    <div class="metric-val mv-red">{counts['badge-low']}</div>
    <div class="metric-label">Low Similarity</div>
  </div>
  <div class="metric-card mc-gray">
    <span class="metric-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg></span>
    <div class="metric-val mv-gray">{counts['badge-no']}</div>
    <div class="metric-label">No Similarity</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Result list
        st.markdown(
            f'<div class="sec-heading">Top {len(results)} Judul Paling Mirip '
            f'<span class="sec-badge-update">Keyword Highlighted</span></div>',
            unsafe_allow_html=True,
        )

        keywords = query_clean.split()

        for i, res in enumerate(results, 1):
            flagged_class = "flagged" if res["badge"] == "badge-high" else ""
            flag_html = (
                '<span class="flag-pill">⚠️ Perlu ditinjau</span>'
                if res["badge"] == "badge-high" else ""
            )
            highlighted = highlight_keywords(res["judul"], keywords)
            bar_pct     = int(res["skor"] * 100)
            color       = bar_color(res["badge"])

            rank_class = ""
            if i == 1: rank_class = "r1"
            elif i == 2: rank_class = "r2"
            elif i == 3: rank_class = "r3"

            badge_html  = f'<span class="{res["badge"]}">{res["label"]}</span>'
            pill_html   = '<span class="flag-pill">Perlu ditinjau</span>' if res["badge"] == "badge-high" else ""
            card_html = (
                f'<div class="result-card {flagged_class}">'
                f'<div class="rank-badge {rank_class}">#{i}</div>'
                f'<div class="result-body">'
                f'<div class="result-title">{highlighted}</div>'
                f'<div class="meta-row">{badge_html}{pill_html}</div>'
                f'</div>'
                f'<div class="score-col">'
                f'<div class="score-num">{res["skor"]:.4f}</div>'
                f'<div class="score-sub">cosine sim.</div>'
                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_pct}%;background:{color};"></div></div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

        # Ekspor 
        st.markdown("---")
        st.markdown(
            '<div class="sec-heading">Ekspor Hasil '
            '<span class="sec-badge-new">Excel &amp; CSV</span></div>',
            unsafe_allow_html=True,
        )

        col_xl, col_csv, col_space = st.columns([1.5, 1.5, 4])

        xlsx_bytes = results_to_excel(results)
        with col_xl:
            st.download_button(
                label="Unduh Excel (.xlsx)",
                data=xlsx_bytes,
                file_name="hasil_kemiripan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        df_csv = pd.DataFrame([
            {"Rank": i+1, "Judul": r["judul"], "Skor": r["skor"], "Kategori": r["label"]}
            for i, r in enumerate(results)
        ])
        with col_csv:
            st.download_button(
                label="Unduh CSV",
                data=df_csv.to_csv(index=False).encode("utf-8"),
                file_name="hasil_kemiripan.csv",
                mime="text/csv",
                use_container_width=True,
            )

elif cari and not query.strip():
    st.warning("Masukkan judul terlebih dahulu!")

else:
    # Empty state awal 
    st.markdown(f"""
<div class="empty-state">
  <div class="empty-icon-wrap">
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.4"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
  </div>
  <div class="empty-title">Mulai Pencarian Similarity</div>
  <div class="empty-sub">
    Ketik judul skripsi di kolom pencarian di atas, lalu klik tombol
    <strong>Cari</strong> untuk mendeteksi kemiripan dengan
    <strong>{len(docs_original)} judul</strong> yang ada di database.
  </div>
  <div class="empty-hint">Gunakan judul lengkap untuk hasil yang lebih akurat</div>
</div>
""", unsafe_allow_html=True)
