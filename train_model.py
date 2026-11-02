"""
train_model.py
==============
Membangun model rekomendasi HYBRID (2 komponen) untuk dataset Book-Crossing:
  1) Latent-factor Collaborative Filtering (PureSVD, implicit + explicit feedback)
  2) Content-Based Filtering (TF-IDF: judul + penulis + penerbit)
Lalu mengevaluasi (leave-one-out) & menyimpan semua artefak untuk dashboard.

Jalankan:  python train_model.py
Letakkan BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv di folder ./data
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Path data: Kaggle jika ada, selain itu folder lokal ./data
_KAGGLE = Path("/kaggle/input/datasets/lailadyah/book-adl")
DATA = _KAGGLE if _KAGGLE.exists() else Path("data")
OUT = Path("/kaggle/working/artifacts") if Path("/kaggle/working").exists() else Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Hyper-parameter ----------------
MIN_USER_INTERACTIONS = 5      # user minimal punya 5 interaksi positif
MIN_ITEM_INTERACTIONS = 5      # buku minimal diinteraksi 5 user
N_FACTORS = 64                 # dimensi laten SVD
POSITIVE_MIN_RATING = 5        # rating eksplisit 1-4 dianggap negatif -> dibuang
IMPLICIT_WEIGHT = 1.0          # bobot rating 0 (interaksi tanpa nilai)
N_EVAL_USERS = 3000
TOP_K = 10
SEED = 42
rng = np.random.default_rng(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================
# 1. LOAD & CLEANING
# =====================================================
def read_bx(name):
    return pd.read_csv(DATA / name, sep=";", encoding="latin-1", quotechar='"',
                       escapechar="\\", on_bad_lines="skip", low_memory=False)


def fix_mojibake(x):
    """File asli UTF-8 yang terbaca sebagai latin-1 (mis. 'cÃ¡mara' -> 'cámara')."""
    try:
        return x.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return x


def load_data():
    users = read_bx("BX-Users.csv")
    books = read_bx("BX-Books.csv")
    ratings = read_bx("BX-Book-Ratings.csv")
    users.columns = ["user_id", "location", "age"]
    books.columns = ["isbn", "title", "author", "year", "publisher",
                     "img_s", "img_m", "img_l"]
    ratings.columns = ["user_id", "isbn", "rating"]

    # --- users
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users.loc[(users["age"] < 5) | (users["age"] > 100), "age"] = np.nan
    loc = users["location"].fillna("").str.split(",")
    users["country"] = loc.str[-1].str.strip().replace("", np.nan)
    users["country"] = users["country"].fillna("unknown")
    users["city"] = loc.str[0].str.strip()

    # --- books
    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    books.loc[(books["year"] < 1900) | (books["year"] > 2026), "year"] = np.nan
    for c in ["title", "author", "publisher"]:
        books[c] = (books[c].fillna("").astype(str).map(fix_mojibake)
                    .str.replace("&amp;", "&", regex=False).str.strip())
    books = books.drop_duplicates("isbn")
    books = books[books["title"] != ""]

    # --- ratings
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating", "user_id"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["isbn"] = ratings["isbn"].astype(str).str.strip()
    ratings = ratings.drop_duplicates(["user_id", "isbn"])
    # hanya buku yang punya metadata (dibutuhkan untuk content-based & tampilan)
    ratings = ratings[ratings["isbn"].isin(books["isbn"])]
    return users, books, ratings


def build_interactions(ratings):
    """rating 0 -> implisit (bobot 1); 1-4 -> negatif (dibuang);
    5-10 -> positif eksplisit dengan bobot 1.33 .. 3.0"""
    r = ratings.copy()
    r = r[(r["rating"] == 0) | (r["rating"] >= POSITIVE_MIN_RATING)]
    r["w"] = np.where(r["rating"] == 0, IMPLICIT_WEIGHT,
                      1.0 + (r["rating"] - 4) / 3.0)
    # filter iteratif
    while True:
        n0 = len(r)
        uc = r["user_id"].value_counts()
        r = r[r["user_id"].isin(uc[uc >= MIN_USER_INTERACTIONS].index)]
        ic = r["isbn"].value_counts()
        r = r[r["isbn"].isin(ic[ic >= MIN_ITEM_INTERACTIONS].index)]
        if len(r) == n0:
            break
    return r.reset_index(drop=True)


# =====================================================
# 2. MODEL
# =====================================================
def make_matrix(df, u2i, i2i):
    rows = df["user_id"].map(u2i).values
    cols = df["isbn"].map(i2i).values
    return sparse.csr_matrix((df["w"].values.astype(np.float32), (rows, cols)),
                             shape=(len(u2i), len(i2i)))


def normalize_users(X):
    """Redam dominasi heavy-user: bagi tiap baris dengan sqrt(jumlah interaksi)."""
    deg = np.asarray((X > 0).sum(1)).ravel().astype(np.float32)
    d = sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1)))
    return d @ X


def fit_svd(X):
    svd = TruncatedSVD(n_components=N_FACTORS, n_iter=7, random_state=SEED)
    svd.fit(normalize_users(X))
    return svd.components_.T.astype(np.float32)  # (n_items, k)


def cf_scores(x, V):
    """x: vektor sparse 1 x n_items (profil user). Fold-in: (x V) V^T."""
    return np.asarray((x @ V) @ V.T).ravel()


def content_scores(x, T):
    prof = x @ T                      # 1 x n_terms
    return np.asarray((prof @ T.T).todense()).ravel()


def minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def hybrid(cf, ct, w):
    """w = (bobot_cf, bobot_content), dijumlah = 1."""
    return w[0] * minmax(cf) + w[1] * minmax(ct)


# =====================================================
# 3. EVALUASI (leave-one-out)
# =====================================================
def evaluate(X, T, w_grid):
    n_users, n_items = X.shape
    eligible = np.where(np.diff(X.indptr) >= 6)[0]
    eval_users = rng.choice(eligible, size=min(N_EVAL_USERS, len(eligible)),
                            replace=False)
    # sembunyikan 1 item positif acak per user eval
    Xtr = X.tolil(copy=True)
    held = {}
    for u in eval_users:
        items = X.indices[X.indptr[u]:X.indptr[u + 1]]
        h = rng.choice(items)
        held[u] = h
        Xtr[u, h] = 0
    Xtr = Xtr.tocsr()
    Xtr.eliminate_zeros()
    V = fit_svd(Xtr)
    log("  SVD evaluasi selesai")

    models = {"CF (SVD)": (1.0, 0.0), "Content-Based": (0.0, 1.0)}
    for w in w_grid:
        models[f"Hybrid {w}"] = w
    res = {m: {"hit": 0, "ndcg": 0, "mrr": 0} for m in models}

    for u in eval_users:
        x = Xtr[u]
        seen = x.indices
        cf = cf_scores(x, V)
        ct = content_scores(x, T)
        for name, w in models.items():
            s = hybrid(cf, ct, w)
            s[seen] = -np.inf
            tgt = s[held[u]]
            rank = int((s > tgt).sum())          # 0-based
            if rank < TOP_K:
                res[name]["hit"] += 1
                res[name]["ndcg"] += 1.0 / np.log2(rank + 2)
            res[name]["mrr"] += 1.0 / (rank + 1)
    n = len(eval_users)
    out = [{"model": m, f"HitRate@{TOP_K}": v["hit"] / n,
            f"NDCG@{TOP_K}": v["ndcg"] / n, "MRR": v["mrr"] / n}
           for m, v in res.items()]
    return out, models


# =====================================================
# MAIN
# =====================================================
def main():
    log("Memuat & membersihkan data ...")
    users, books, ratings = load_data()
    log(f"  users={len(users):,} books={len(books):,} ratings={len(ratings):,}")

    # ---- statistik EDA (sebelum filter) untuk dashboard
    eda = {
        "n_users": int(len(users)), "n_books": int(len(books)),
        "n_ratings": int(len(ratings)),
        "pct_implicit": float((ratings["rating"] == 0).mean()),
        "rating_dist": ratings["rating"].astype(int).value_counts()
                       .sort_index().to_dict(),
        "sparsity": float(1 - len(ratings) / (ratings["user_id"].nunique()
                                              * ratings["isbn"].nunique())),
    }
    explicit = ratings[ratings["rating"] > 0]
    eda["n_explicit"] = int(len(explicit))

    inter = build_interactions(ratings)
    log(f"  interaksi positif setelah filter: {len(inter):,} | "
        f"user={inter.user_id.nunique():,} | buku={inter.isbn.nunique():,}")

    u_ids = np.sort(inter["user_id"].unique())
    i_ids = np.sort(inter["isbn"].unique())
    u2i = {u: k for k, u in enumerate(u_ids)}
    i2i = {i: k for k, i in enumerate(i_ids)}
    X = make_matrix(inter, u2i, i2i)
    eda["model_users"], eda["model_items"], eda["model_inter"] = \
        len(u_ids), len(i_ids), int(X.nnz)

    # ---- tabel item
    items = books.set_index("isbn").loc[i_ids].reset_index()
    ex = explicit.groupby("isbn")["rating"].agg(["mean", "count"])
    items = items.merge(ex, left_on="isbn", right_index=True, how="left")
    items["n_inter"] = np.asarray((X > 0).sum(0)).ravel()
    items["count"] = items["count"].fillna(0)
    gm = explicit["rating"].mean()
    m_prior = 10
    items["avg_rating"] = items["mean"]
    items["bayes_rating"] = ((items["count"] * items["mean"].fillna(gm)
                              + m_prior * gm) / (items["count"] + m_prior))
    items = items.rename(columns={"count": "n_explicit"})
    items = items[["isbn", "title", "author", "year", "publisher", "img_m",
                   "n_inter", "n_explicit", "avg_rating", "bayes_rating"]]

    # ---- content TF-IDF
    log("TF-IDF konten ...")
    text = (items["title"] + " " + items["author"] + " " + items["author"]
            + " " + items["publisher"]).str.lower()
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2),
                            max_features=60000, sublinear_tf=True)
    T = normalize(tfidf.fit_transform(text)).astype(np.float32).tocsr()
    log(f"  TF-IDF matrix {T.shape}")

    # ---- evaluasi
    log("Evaluasi leave-one-out ...")
    w_grid = [(round(float(a), 1), round(float(1 - a), 1)) for a in np.arange(0.1, 1.0, 0.1)]
    metrics, models = evaluate(X, T, w_grid)
    mdf = pd.DataFrame(metrics)
    print(mdf.round(4).to_string(index=False))
    best = mdf[mdf.model.str.startswith("Hybrid")].sort_values(
        f"NDCG@{TOP_K}", ascending=False).iloc[0]["model"]
    best_w = models[best]
    log(f"Bobot hybrid terbaik: {best}")

    # ---- model final (seluruh data)
    log("Fit SVD final ...")
    V = fit_svd(X)

    # ---- tabel user + cold-start populer per negara
    ut = users.set_index("user_id").reindex(u_ids)
    ut["n_inter"] = np.diff(X.indptr)
    ut = ut.reset_index()[["user_id", "country", "city", "age", "n_inter"]]
    inter_c = inter.merge(users[["user_id", "country"]], on="user_id")
    country_pop = (inter_c.groupby(["country", "isbn"])["w"].sum()
                   .reset_index())
    top_c = inter_c["country"].value_counts().head(25).index
    country_pop = country_pop[country_pop.country.isin(top_c)]

    # ---- EDA tambahan
    eda["age_hist"] = users["age"].dropna().astype(int).value_counts() \
        .sort_index().to_dict()
    eda["top_countries"] = users["country"].value_counts().head(15).to_dict()
    eda["year_hist"] = books["year"].dropna().astype(int).value_counts() \
        .sort_index().to_dict()
    eda["top_authors"] = inter.merge(items[["isbn", "author"]], on="isbn") \
        ["author"].value_counts().head(15).to_dict()

    joblib.dump({
        "X": X, "V": V, "T": T,
        "items": items, "users": ut, "u2i": u2i, "i2i": i2i,
        "u_ids": u_ids, "i_ids": i_ids, "best_w": best_w,
        "country_pop": country_pop, "eda": eda,
        "metrics": mdf, "top_k": TOP_K,
    }, OUT / "model.joblib", compress=3)
    json.dump(eda, open(OUT / "eda.json", "w"), default=str)
    log("Selesai -> artifacts/model.joblib")


if __name__ == "__main__":
    main()
