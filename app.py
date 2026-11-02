"""
app.py - Dashboard Sistem Rekomendasi Buku (Book-Crossing)
Jalankan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import recsys

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Memuat model ...")
def get_art():
    return recsys.load_artifacts("model.joblib")


try:
    art = get_art()
except FileNotFoundError:
    st.error("File artifacts/model.joblib belum ada. Jalankan dulu: "
             "`python train_model.py`")
    st.stop()

items, users, eda, metrics = art["items"], art["users"], art["eda"], art["metrics"]
K_NAME = f"HitRate@{art['top_k']}"
N_NAME = f"NDCG@{art['top_k']}"


# ---------------- util tampilan ----------------
def book_cards(df, cols=5, extra=None):
    """Tampilkan buku sebagai kartu bergambar."""
    for start in range(0, len(df), cols):
        row = st.columns(cols)
        for c, (_, b) in zip(row, df.iloc[start:start + cols].iterrows()):
            with c:
                img = b["img_m"] if isinstance(b["img_m"], str) else None
                if img:
                    st.image(img, use_container_width=True)
                st.markdown(f"**{b['title'][:70]}**")
                yr = f"{int(b['year'])}" if pd.notna(b["year"]) else "-"
                st.caption(f"{b['author']} · {yr}")
                rt = b["avg_rating"]
                st.caption(f"⭐ {rt:.1f}" if pd.notna(rt) else "⭐ n/a")
                if extra and extra in b and pd.notna(b[extra]):
                    st.caption(str(b[extra]))


def result_table(df):
    t = df[["title", "author", "year", "publisher", "score",
            "s_cf", "s_content", "reason"]].copy()
    t.columns = ["Judul", "Penulis", "Tahun", "Penerbit", "Skor Hybrid",
                 "CF", "Konten", "Alasan"]
    st.dataframe(t, use_container_width=True, hide_index=True)


# ---------------- sidebar ----------------
st.sidebar.title("📚 Book Recommender")
st.sidebar.caption("Hybrid: Collaborative Filtering (SVD) + Content-Based "
                   "(TF-IDF)")
st.sidebar.markdown("### Pengaturan")
k = st.sidebar.slider("Jumlah rekomendasi", 5, 20, 10)
max_auth = st.sidebar.slider("Maks. buku per penulis", 1, 5, 2)
yr_valid = items["year"].dropna()
yr_rng = st.sidebar.slider("Rentang tahun terbit",
                           int(yr_valid.min()), int(yr_valid.max()),
                           (int(yr_valid.min()), int(yr_valid.max())))
st.sidebar.markdown("### Bobot Hybrid")
bw = art["best_w"]
use_custom = st.sidebar.checkbox("Atur bobot manual", value=False)
if use_custom:
    w_cf = st.sidebar.slider("Bobot Collaborative (CF)", 0.0, 1.0,
                             float(bw[0]), 0.05)
    weights = (w_cf, round(1 - w_cf, 2))
    st.sidebar.caption(f"Bobot Content-Based otomatis = {weights[1]}")
else:
    weights = bw
    st.sidebar.info(f"Bobot terbaik (hasil evaluasi): CF={bw[0]}, "
                    f"Konten={bw[1]}")
rec_kw = dict(k=k, weights=weights, year_min=yr_rng[0], year_max=yr_rng[1],
              max_per_author=max_auth)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan Data", "👤 Rekomendasi untuk User",
    "🆕 User Baru (pilih buku)", "🔎 Buku Serupa", "🧪 Evaluasi Model"])

# =========================================================
# TAB 1 : EDA
# =========================================================
with tab1:
    st.subheader("Ringkasan Dataset")
    c = st.columns(5)
    c[0].metric("Total User", f"{eda['n_users']:,}")
    c[1].metric("Total Buku", f"{eda['n_books']:,}")
    c[2].metric("Total Rating", f"{eda['n_ratings']:,}")
    c[3].metric("Rating Implisit (0)", f"{eda['pct_implicit']*100:.0f}%")
    c[4].metric("Sparsity", f"{eda['sparsity']*100:.3f}%")
    st.caption(f"Data yang dipakai model setelah filtering: "
               f"{eda['model_users']:,} user × {eda['model_items']:,} buku "
               f"({eda['model_inter']:,} interaksi positif).")

    a, b = st.columns(2)
    rd = pd.Series(eda["rating_dist"]).rename_axis("Rating").reset_index(name="Jumlah")
    a.plotly_chart(px.bar(rd, x="Rating", y="Jumlah",
                          title="Distribusi Rating (0 = implisit)"),
                   use_container_width=True)
    ah = pd.Series(eda["age_hist"]).rename_axis("Usia").reset_index(name="Jumlah")
    ah = ah[(ah["Usia"] >= 5) & (ah["Usia"] <= 90)]
    b.plotly_chart(px.bar(ah, x="Usia", y="Jumlah",
                          title="Distribusi Usia User (data tersedia)"),
                   use_container_width=True)

    a, b = st.columns(2)
    tc = pd.Series(eda["top_countries"]).rename_axis("Negara").reset_index(name="User")
    a.plotly_chart(px.bar(tc, x="User", y="Negara", orientation="h",
                          title="15 Negara dengan User Terbanyak")
                   .update_yaxes(autorange="reversed"), use_container_width=True)
    ta = pd.Series(eda["top_authors"]).rename_axis("Penulis").reset_index(name="Interaksi")
    b.plotly_chart(px.bar(ta, x="Interaksi", y="Penulis", orientation="h",
                          title="15 Penulis Terpopuler")
                   .update_yaxes(autorange="reversed"), use_container_width=True)

    yh = pd.Series(eda["year_hist"]).rename_axis("Tahun").reset_index(name="Buku")
    yh = yh[yh["Tahun"] >= 1950]
    st.plotly_chart(px.area(yh, x="Tahun", y="Buku",
                            title="Jumlah Buku per Tahun Terbit"),
                    use_container_width=True)

    st.subheader("🔥 Buku Paling Banyak Dibaca (statistik, bukan komponen model)")
    ctry = ["(Semua negara)"] + sorted(art["country_pop"]["country"].unique())
    sel = st.selectbox("Filter negara", ctry)
    pop = recsys.popular_by_country(art, sel, 10)
    book_cards(pop, cols=5)

# =========================================================
# TAB 2 : user existing
# =========================================================
with tab2:
    st.subheader("Rekomendasi untuk User yang Sudah Ada")
    c1, c2 = st.columns([1, 3])
    with c1:
        min_i = st.number_input("Min. jumlah interaksi", 5, 500, 10)
        pool = users[users["n_inter"] >= min_i]
        if st.button("🎲 User acak"):
            st.session_state["uid"] = int(pool.sample(1).user_id.iloc[0])
        uid = st.number_input("User ID", value=st.session_state.get(
            "uid", int(pool.user_id.iloc[0])), step=1)
    x = recsys.user_vector(art, uid)
    if x is None:
        st.warning("User ID tidak ada di data model (user belum punya cukup "
                   "interaksi). Gunakan tab 'User Baru' atau lihat buku populer.")
    else:
        u = users[users.user_id == uid].iloc[0]
        with c2:
            m = st.columns(4)
            m[0].metric("User ID", int(uid))
            m[1].metric("Negara", str(u["country"]).title())
            m[2].metric("Usia", int(u["age"]) if pd.notna(u["age"]) else "n/a")
            m[3].metric("Buku diinteraksi", int(u["n_inter"]))
        hist = items.iloc[x.indices].copy()
        hist["w"] = x.data
        hist = hist.sort_values("w", ascending=False)
        with st.expander(f"📖 Riwayat buku user ({len(hist)}) — 10 teratas"):
            book_cards(hist.head(10), cols=5)
        st.markdown("### ✨ Rekomendasi")
        rec = recsys.recommend(art, x, **rec_kw)
        if rec.empty:
            st.info("Tidak ada hasil dengan filter saat ini.")
        else:
            book_cards(rec, cols=5, extra="reason")
            with st.expander("Detail skor & alasan"):
                result_table(rec)
                comp = rec.set_index("title")[["s_cf", "s_content"]]
                comp.columns = ["CF", "Konten"]
                st.bar_chart(comp)

# =========================================================
# TAB 3 : new user (cold-start)
# =========================================================
with tab3:
    st.subheader("User Baru — pilih buku yang kamu suka")
    st.caption("Cari judul/penulis, tambahkan ke daftar, beri rating, "
               "lalu dapatkan rekomendasi.")
    if "picks" not in st.session_state:
        st.session_state["picks"] = {}
    q = st.text_input("Cari judul atau penulis", placeholder="mis. harry potter, grisham")
    res = recsys.search_books(art, q)
    if len(res):
        opts = {f"{r.title[:70]} — {r.author}": r.isbn for r in res.itertuples()}
        ch = st.selectbox("Hasil pencarian", list(opts))
        rt = st.slider("Rating kamu (0 = hanya pernah baca)", 0, 10, 8)
        if st.button("➕ Tambah ke daftar"):
            st.session_state["picks"][opts[ch]] = rt
    picks = st.session_state["picks"]
    if picks:
        lib = items.set_index("isbn")
        st.markdown("**Daftar bukumu:**")
        for isbn, r in list(picks.items()):
            a, b = st.columns([6, 1])
            a.write(f"• {lib.at[isbn, 'title']} — {lib.at[isbn, 'author']} (⭐ {r})")
            if b.button("🗑", key=f"del{isbn}"):
                del st.session_state["picks"][isbn]
                st.rerun()
        xv = recsys.vector_from_isbns(art, picks)
        st.markdown("### ✨ Rekomendasi untukmu")
        rec = recsys.recommend(art, xv, **rec_kw)
        if not rec.empty:
            book_cards(rec, cols=5, extra="reason")
            with st.expander("Detail skor"):
                result_table(rec)
    else:
        st.info("Belum ada buku. Mulai dengan mencari buku di atas. "
                "Tanpa riwayat, sistem hanya bisa menampilkan buku populer "
                "(lihat tab Ringkasan Data).")

# =========================================================
# TAB 4 : similar books
# =========================================================
with tab4:
    st.subheader("Cari Buku Serupa (Item-to-Item)")
    q2 = st.text_input("Judul/penulis buku acuan", key="q2",
                       placeholder="mis. the firm")
    r2 = recsys.search_books(art, q2)
    if len(r2):
        opts2 = {f"{r.title[:70]} — {r.author}": r.isbn for r in r2.itertuples()}
        ch2 = st.selectbox("Pilih buku", list(opts2), key="ch2")
        alpha = st.slider("Bobot kemiripan perilaku pembaca (CF) vs konten",
                          0.0, 1.0, 0.3, 0.05)
        sim = recsys.similar_books(art, opts2[ch2], k=k, alpha=alpha)
        book_cards(sim, cols=5)
    else:
        st.info("Ketik minimal 2 huruf untuk mencari buku.")

# =========================================================
# TAB 5 : evaluasi
# =========================================================
with tab5:
    st.subheader("Evaluasi Model (Leave-One-Out)")
    st.caption("Satu buku positif milik tiap user disembunyikan; model diminta "
               f"menebaknya di antara {eda['model_items']:,} buku. "
               "Makin tinggi makin baik.")
    show = metrics.copy()
    st.dataframe(show.style.format({K_NAME: "{:.2%}", N_NAME: "{:.4f}",
                                    "MRR": "{:.4f}"})
                 .highlight_max(subset=[K_NAME, N_NAME, "MRR"], color="#c6efce"),
                 use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(show, x="model", y=K_NAME,
                           title=f"{K_NAME} per Model"),
                    use_container_width=True)
    st.markdown(
        "**Interpretasi:** model hybrid dibandingkan dengan CF murni dan "
        "content-based murni. CF menangkap pola perilaku pembaca, sedangkan "
        "content-based menangkap kesamaan penulis/judul/penerbit; "
        "penggabungannya saling menutup kelemahan masing-masing.")
