# 📖 Aksara Match — Smart Book Recommendations

Hybrid recommender (Collaborative Filtering SVD + Content-Based TF-IDF) untuk dataset Book-Crossing,
dengan dashboard bertema *Dark Library* (gaya CineMatch).

## Struktur
```
bookrec/
├── data/                  <- taruh BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv (hanya jika melatih ulang)
├── artifacts/model.joblib <- model hasil training (sudah disertakan)
├── train_model.py         <- preprocessing + training + evaluasi
├── recsys.py              <- logika rekomendasi
├── app.py                 <- dashboard Streamlit "Aksara Match"
├── book_recommender_hybrid.ipynb  <- notebook (Kaggle) versi lengkap
└── requirements.txt
```

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```
Melatih ulang (opsional): `python train_model.py` (± 1 menit).

## Halaman
- 🏠 **Home** — hero, statistik, cara kerja, buku rating tertinggi
- 📊 **Analytics** — distribusi rating, negara/penulis/usia, tahun terbit, evaluasi model
- 🎯 **Recommend** — berdasarkan satu buku yang disukai, atau berdasarkan Reader ID
- 📚 **Browse Library** — filter penulis / penerbit / dekade
- Halaman **detail buku** (tombol *View Details*) + "More Like This"
