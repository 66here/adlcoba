"""recsys.py - logika inferensi model hybrid (dipakai oleh dashboard)."""
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


def load_artifacts(path="artifacts/model.joblib"):
    art = joblib.load(path)
    art["Vn"] = normalize(art["V"])          # untuk kemiripan item (CF)
    return art


def _norm_title(t):
    """Normalisasi judul agar edisi berbeda dari buku sama dianggap satu."""
    import re
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", str(t).lower())
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


# ---------- profil user ----------
def user_vector(art, user_id):
    """Vektor interaksi 1 x n_items untuk user yang ada di model."""
    u = art["u2i"].get(int(user_id))
    return None if u is None else art["X"][u]


def vector_from_isbns(art, picks):
    """picks: dict {isbn: rating(0-10)} -> vektor interaksi (user baru)."""
    n = art["X"].shape[1]
    idx, val = [], []
    for isbn, r in picks.items():
        j = art["i2i"].get(isbn)
        if j is None:
            continue
        idx.append(j)
        val.append(1.0 if r == 0 else 1.0 + (max(r, 5) - 4) / 3.0)
    if not idx:
        return None
    return sparse.csr_matrix((val, ([0] * len(idx), idx)), shape=(1, n),
                             dtype=np.float32)


# ---------- skor ----------
def component_scores(art, x):
    """Kembalikan skor CF dan content (masing-masing array n_items)."""
    V, T = art["V"], art["T"]
    # normalisasi seperti saat training (bagi sqrt(jumlah interaksi))
    deg = max(int((x > 0).sum()), 1)
    xn = x / np.sqrt(deg)
    cf = np.asarray((xn @ V) @ V.T).ravel()
    ct = np.asarray(((x @ T) @ T.T).todense()).ravel()
    return cf, ct


def recommend(art, x, k=10, weights=None, year_min=None, year_max=None,
              max_per_author=2, extra_exclude=None):
    """Top-K rekomendasi hybrid (CF + content) + skor komponen & alasan."""
    w = weights or art["best_w"]
    cf, ct = component_scores(art, x)
    cf_n, ct_n = _minmax(cf), _minmax(ct)
    score = w[0] * cf_n + w[1] * ct_n

    items = art["items"]
    mask = np.zeros(len(items), bool)
    mask[x.indices] = True                      # sudah dibaca
    if extra_exclude is not None:
        mask[list(extra_exclude)] = True
    if year_min is not None:
        mask |= (items["year"].fillna(year_min).values < year_min)
    if year_max is not None:
        mask |= (items["year"].fillna(year_max).values > year_max)
    score = np.where(mask, -np.inf, score)

    order = np.argsort(-score)[: k * 15]
    picked, per_author = [], {}
    seen_titles = {_norm_title(t) for t in items["title"].values[x.indices]}
    for j in order:
        if not np.isfinite(score[j]):
            break
        a = items.at[j, "author"]
        nt = _norm_title(items.at[j, "title"])
        if per_author.get(a, 0) >= max_per_author or nt in seen_titles:
            continue
        seen_titles.add(nt)
        per_author[a] = per_author.get(a, 0) + 1
        picked.append(j)
        if len(picked) == k:
            break
    picked = np.array(picked, dtype=int)
    if len(picked) == 0:
        return pd.DataFrame()

    res = items.iloc[picked].copy()
    res["score"] = score[picked]
    res["s_cf"], res["s_content"] = cf_n[picked], ct_n[picked]
    res["reason"] = _explain(art, x, picked)
    return res.reset_index(drop=True)


def _explain(art, x, picked):
    """Cari buku riwayat yang paling mirip tiap rekomendasi (alasan)."""
    hist = x.indices
    if len(hist) == 0:
        return ["-"] * len(picked)
    Vn, T, items = art["Vn"], art["T"], art["items"]
    sim = 0.5 * (Vn[picked] @ Vn[hist].T) \
        + 0.5 * np.asarray((T[picked] @ T[hist].T).todense())
    out = []
    for r in range(len(picked)):
        h = hist[int(np.argmax(sim[r]))]
        out.append(f"Similar to «{items.at[h, 'title'][:60]}»")
    return out


def similar_books(art, isbn, k=10, alpha=0.3):
    j = art["i2i"].get(isbn)
    if j is None:
        return pd.DataFrame()
    Vn, T = art["Vn"], art["T"]
    s_cf = Vn @ Vn[j]
    s_ct = np.asarray((T @ T[j].T).todense()).ravel()
    s = alpha * _minmax(s_cf) + (1 - alpha) * _minmax(s_ct)
    s[j] = -np.inf
    items = art["items"]
    base = _norm_title(items.at[j, "title"])
    top, seen = [], {base}
    for c in np.argsort(-s)[: k * 15]:
        nt = _norm_title(items.at[c, "title"])
        if nt in seen:
            continue
        seen.add(nt)
        top.append(c)
        if len(top) == k:
            break
    top = np.array(top, dtype=int)
    res = items.iloc[top].copy()
    res["similarity"] = s[top]
    return res.reset_index(drop=True)


def popular_by_country(art, country=None, k=10):
    items = art["items"]
    if country and country != "(Semua negara)":
        cp = art["country_pop"]
        cp = cp[cp.country == country].nlargest(k * 3, "w")
        res = items.merge(cp[["isbn", "w"]], on="isbn")
        res = res.sort_values("w", ascending=False)
        return res.head(k).reset_index(drop=True)
    res = items.assign(w=items["n_inter"]).nlargest(k, "w")
    return res.reset_index(drop=True)


def search_books(art, query, limit=30):
    if not query or len(query.strip()) < 2:
        return pd.DataFrame()
    q = query.lower().strip()
    it = art["items"]
    m = it["title"].str.lower().str.contains(q, regex=False) | \
        it["author"].str.lower().str.contains(q, regex=False)
    return it[m].sort_values("n_inter", ascending=False).head(limit)
