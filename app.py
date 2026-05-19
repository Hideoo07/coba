"""
Backend API - Sistem Rekomendasi Properti
Menggunakan Flask + AHP + Profile Matching
+ Penyesuaian Wilayah: Provinsi → Kota → Kecamatan
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ===================== MAPPING PROVINSI =====================
# Daftar kota/kabupaten beserta provinsinya
# Tambahkan kota baru di sini jika dataset bertambah
PROVINSI_KOTA_MAP = {
    "Jawa Barat": [
        "Bandung", "Kota Bandung", "Kabupaten Bandung", "Kabupaten Bandung Barat",
        "Bekasi", "Kota Bekasi", "Kabupaten Bekasi",
        "Bogor", "Kota Bogor", "Kabupaten Bogor",
        "Cimahi", "Kota Cimahi",
        "Cianjur", "Kabupaten Cianjur",
        "Cirebon", "Kota Cirebon", "Kabupaten Cirebon",
        "Depok", "Kota Depok",
        "Garut", "Kabupaten Garut",
        "Indramayu", "Kabupaten Indramayu",
        "Karawang", "Kabupaten Karawang",
        "Kuningan", "Kabupaten Kuningan",
        "Majalengka", "Kabupaten Majalengka",
        "Pangandaran", "Kabupaten Pangandaran",
        "Purwakarta", "Kabupaten Purwakarta",
        "Subang", "Kabupaten Subang",
        "Sukabumi", "Kota Sukabumi", "Kabupaten Sukabumi",
        "Sumedang", "Kabupaten Sumedang",
        "Tasikmalaya", "Kota Tasikmalaya", "Kabupaten Tasikmalaya",
    ],
    "Jawa Tengah": [
        "Banjarnegara", "Kabupaten Banjarnegara",
        "Banyumas", "Kabupaten Banyumas",
        "Batang", "Kabupaten Batang",
        "Blora", "Kabupaten Blora",
        "Boyolali", "Kabupaten Boyolali",
        "Brebes", "Kabupaten Brebes",
        "Cilacap", "Kabupaten Cilacap",
        "Demak", "Kabupaten Demak",
        "Grobogan", "Kabupaten Grobogan",
        "Jepara", "Kabupaten Jepara",
        "Karanganyar", "Kabupaten Karanganyar",
        "Kebumen", "Kabupaten Kebumen",
        "Kendal", "Kabupaten Kendal",
        "Klaten", "Kabupaten Klaten",
        "Kudus", "Kabupaten Kudus",
        "Magelang", "Kota Magelang", "Kabupaten Magelang",
        "Pati", "Kabupaten Pati",
        "Pekalongan", "Kota Pekalongan", "Kabupaten Pekalongan",
        "Pemalang", "Kabupaten Pemalang",
        "Purbalingga", "Kabupaten Purbalingga",
        "Purworejo", "Kabupaten Purworejo",
        "Rembang", "Kabupaten Rembang",
        "Salatiga", "Kota Salatiga",
        "Semarang", "Kota Semarang", "Kabupaten Semarang",
        "Sragen", "Kabupaten Sragen",
        "Sukoharjo", "Kabupaten Sukoharjo",
        "Surakarta", "Kota Surakarta", "Solo", "Kota Solo",
        "Tegal", "Kota Tegal", "Kabupaten Tegal",
        "Temanggung", "Kabupaten Temanggung",
        "Wonogiri", "Kabupaten Wonogiri",
        "Wonosobo", "Kabupaten Wonosobo",
    ],
    "Jawa Timur": [
        "Bangkalan", "Kabupaten Bangkalan",
        "Banyuwangi", "Kabupaten Banyuwangi",
        "Blitar", "Kota Blitar", "Kabupaten Blitar",
        "Bojonegoro", "Kabupaten Bojonegoro",
        "Bondowoso", "Kabupaten Bondowoso",
        "Gresik", "Kabupaten Gresik",
        "Jember", "Kabupaten Jember",
        "Jombang", "Kabupaten Jombang",
        "Kediri", "Kota Kediri", "Kabupaten Kediri",
        "Lamongan", "Kabupaten Lamongan",
        "Lumajang", "Kabupaten Lumajang",
        "Madiun", "Kota Madiun", "Kabupaten Madiun",
        "Magetan", "Kabupaten Magetan",
        "Malang", "Kota Malang", "Kabupaten Malang",
        "Mojokerto", "Kota Mojokerto", "Kabupaten Mojokerto",
        "Nganjuk", "Kabupaten Nganjuk",
        "Ngawi", "Kabupaten Ngawi",
        "Pacitan", "Kabupaten Pacitan",
        "Pamekasan", "Kabupaten Pamekasan",
        "Pasuruan", "Kota Pasuruan", "Kabupaten Pasuruan",
        "Ponorogo", "Kabupaten Ponorogo",
        "Probolinggo", "Kota Probolinggo", "Kabupaten Probolinggo",
        "Sampang", "Kabupaten Sampang",
        "Sidoarjo", "Kabupaten Sidoarjo",
        "Situbondo", "Kabupaten Situbondo",
        "Sumenep", "Kabupaten Sumenep",
        "Surabaya", "Kota Surabaya",
        "Trenggalek", "Kabupaten Trenggalek",
        "Tuban", "Kabupaten Tuban",
        "Tulungagung", "Kabupaten Tulungagung",
    ],
    "DKI Jakarta": [
        "Jakarta Pusat", "Kota Jakarta Pusat",
        "Jakarta Selatan", "Kota Jakarta Selatan",
        "Jakarta Timur", "Kota Jakarta Timur",
        "Jakarta Utara", "Kota Jakarta Utara",
        "Jakarta Barat", "Kota Jakarta Barat",
        "Kepulauan Seribu", "Kabupaten Kepulauan Seribu",
    ],
    "Banten": [
        "Cilegon", "Kota Cilegon",
        "Lebak", "Kabupaten Lebak",
        "Pandeglang", "Kabupaten Pandeglang",
        "Serang", "Kota Serang", "Kabupaten Serang",
        "Tangerang", "Kota Tangerang", "Kabupaten Tangerang",
        "Tangerang Selatan", "Kota Tangerang Selatan",
    ],
    "DI Yogyakarta": [
        "Bantul", "Kabupaten Bantul",
        "Gunungkidul", "Kabupaten Gunungkidul",
        "Kulon Progo", "Kabupaten Kulon Progo",
        "Sleman", "Kabupaten Sleman",
        "Yogyakarta", "Kota Yogyakarta",
    ],
}

# Buat lookup terbalik: kota → provinsi (lowercase untuk matching)
KOTA_TO_PROVINSI = {}
for provinsi, kota_list in PROVINSI_KOTA_MAP.items():
    for kota in kota_list:
        KOTA_TO_PROVINSI[kota.lower()] = provinsi


def get_provinsi(kota_kab):
    """Cari provinsi berdasarkan nama kota/kabupaten."""
    if pd.isna(kota_kab):
        return "Lainnya"
    return KOTA_TO_PROVINSI.get(str(kota_kab).strip().lower(), "Lainnya")


# ===================== LOAD DATASET =====================
DATA_PATH = "data_original.xlsx"


def load_data():
    """Load dan preprocessing dataset properti"""
    if not os.path.exists(DATA_PATH):
        return generate_sample_data()
    df = pd.read_excel(DATA_PATH)
    df = preprocess(df)
    return df


def generate_sample_data():
    """Sample data jika file Excel belum tersedia"""
    np.random.seed(42)
    n = 100
    data_wilayah = [
        ("Menteng", "Jakarta Pusat"),
        ("Kebayoran Baru", "Jakarta Selatan"),
        ("Kelapa Gading", "Jakarta Utara"),
        ("Cibubur", "Jakarta Timur"),
        ("Serpong", "Tangerang Selatan"),
        ("Ciputat", "Tangerang Selatan"),
        ("Margonda", "Depok"),
        ("Cimanggis", "Depok"),
        ("Bekasi Timur", "Bekasi"),
        ("Bekasi Barat", "Bekasi"),
        ("Bogor Tengah", "Bogor"),
        ("Cibinong", "Bogor"),
        ("Cicendo", "Bandung"),
        ("Coblong", "Bandung"),
        ("Lowokwaru", "Malang"),
        ("Klojen", "Malang"),
        ("Gubeng", "Surabaya"),
        ("Rungkut", "Surabaya"),
        ("Tembalang", "Semarang"),
        ("Banyumanik", "Semarang"),
        ("Banjarsari", "Surakarta"),
        ("Laweyan", "Surakarta"),
        ("Depok", "Sleman"),
        ("Gamping", "Sleman"),
    ]
    import random
    rows = []
    for _ in range(n):
        kec, kota = random.choice(data_wilayah)
        rows.append({
            'Price_Clean': np.random.randint(300_000_000, 5_000_000_000),
            'Luas_bangunan': np.random.randint(36, 500),
            'Luas_tanah': np.random.randint(60, 600),
            'Kamar_tidur_clean': np.random.choice([1, 2, 3, 4, 5]),
            'Kamar_mandi': np.random.choice([1, 2, 3, 4]),
            'Jenis_Properti': np.random.choice(['Rumah', 'Apartemen', 'Townhouse']),
            'Kecamatan': kec,
            'Kota_Kab': kota,
        })
    return pd.DataFrame(rows)


def preprocess(df):
    """Bersihkan, siapkan data, dan tambahkan kolom Provinsi otomatis"""
    cols_needed = ['Price_Clean', 'Luas_bangunan', 'Kamar_tidur_clean',
                   'Jenis_Properti', 'Kecamatan', 'Kota_Kab']
    for col in cols_needed:
        if col not in df.columns:
            df[col] = 0
    df = df.dropna(subset=['Price_Clean', 'Luas_bangunan', 'Kamar_tidur_clean'])

    # Tambahkan kolom Provinsi otomatis berdasarkan Kota_Kab
    if 'Provinsi' not in df.columns:
        df['Provinsi'] = df['Kota_Kab'].apply(get_provinsi)
    else:
        # Jika kolom Provinsi sudah ada tapi ada yang kosong, isi otomatis
        mask_kosong = df['Provinsi'].isna() | (df['Provinsi'] == '')
        df.loc[mask_kosong, 'Provinsi'] = df.loc[mask_kosong, 'Kota_Kab'].apply(get_provinsi)

    return df


# Load data saat startup
df_properti = load_data()

# ===================== AHP FUNCTIONS =====================
def ahp_weights(criteria_matrix):
    matrix = np.array(criteria_matrix, dtype=float)
    n = matrix.shape[0]
    geo_means = np.prod(matrix, axis=1) ** (1.0 / n)
    weights = geo_means / geo_means.sum()
    weighted_sum = matrix @ weights
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    return {
        'weights': weights.tolist(),
        'consistency_ratio': round(cr, 4),
        'is_consistent': cr < 0.10
    }


# ===================== PROFILE MATCHING =====================
def profile_matching(df, preferences, weights):
    results = []
    for idx, row in df.iterrows():
        scores = {}

        # GAP Harga
        if preferences.get('max_price', 0) > 0:
            price_ratio = row['Price_Clean'] / preferences['max_price']
            if price_ratio <= 1:
                scores['harga'] = 5 - (1 - price_ratio) * 4
            else:
                scores['harga'] = max(1, 5 - (price_ratio - 1) * 5)
        else:
            scores['harga'] = 3

        # GAP Luas Bangunan
        if preferences.get('min_luas', 0) > 0:
            luas_ratio = row['Luas_bangunan'] / preferences['min_luas']
            scores['luas'] = min(5, max(1, luas_ratio * 3))
        else:
            scores['luas'] = 3

        # GAP Kamar Tidur
        pref_kamar = preferences.get('kamar_tidur', 0)
        if pref_kamar > 0:
            gap_kamar = row['Kamar_tidur_clean'] - pref_kamar
            gap_score_map = {0: 5, 1: 4.5, -1: 4, 2: 3.5, -2: 3}
            scores['kamar'] = gap_score_map.get(gap_kamar, max(1, 3 - abs(gap_kamar)))
        else:
            scores['kamar'] = 3

        # GAP Jenis Properti
        if preferences.get('jenis_properti'):
            scores['jenis'] = 5 if row['Jenis_Properti'] == preferences['jenis_properti'] else 2
        else:
            scores['jenis'] = 3

        # GAP Lokasi — sekarang bertingkat: Provinsi → Kota → Kecamatan
        loc_score = 1  # default: tidak match sama sekali
        row_provinsi = str(row.get('Provinsi', '')).strip().lower()
        row_kota = str(row.get('Kota_Kab', '')).strip().lower()
        row_kec = str(row.get('Kecamatan', '')).strip().lower()

        pref_provinsi = preferences.get('provinsi', '').strip().lower()
        pref_kota = preferences.get('kota', '').strip().lower()
        pref_kec = preferences.get('kecamatan', '').strip().lower()

        if pref_kec and row_kec == pref_kec:
            loc_score = 5          # Match kecamatan (paling spesifik)
        elif pref_kota and row_kota == pref_kota:
            loc_score = 4          # Match kota/kabupaten
        elif pref_provinsi and row_provinsi == pref_provinsi:
            loc_score = 3          # Match provinsi saja
        elif not pref_provinsi and not pref_kota and not pref_kec:
            loc_score = 3          # User tidak pilih lokasi → netral
        # else loc_score tetap 1 (beda provinsi/kota)

        scores['lokasi'] = loc_score

        # Skor akhir dengan bobot AHP
        criteria_keys = ['harga', 'luas', 'kamar', 'jenis', 'lokasi']
        final_score = sum(scores.get(k, 3) * weights[i] for i, k in enumerate(criteria_keys))

        results.append({
            'index': int(idx),
            'scores': scores,
            'final_score': round(final_score, 4)
        })

    return results


# ===================== EVALUATION METRICS =====================
def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant) if relevant else 0

def f1_at_k(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, idx in enumerate(recommended[:k]) if idx in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def mrr(recommended, relevant):
    for i, idx in enumerate(recommended):
        if idx in relevant:
            return 1.0 / (i + 1)
    return 0


# ===================== API ROUTES =====================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/data-info', methods=['GET'])
def data_info():
    """Info dataset + struktur wilayah bertingkat: provinsi → kota → kecamatan"""

    # Bangun struktur wilayah bertingkat
    wilayah = {}
    for _, row in df_properti[['Provinsi', 'Kota_Kab', 'Kecamatan']].dropna().iterrows():
        prov = str(row['Provinsi']).strip()
        kota = str(row['Kota_Kab']).strip()
        kec  = str(row['Kecamatan']).strip()
        if not prov or prov == 'nan':
            continue
        wilayah.setdefault(prov, {})
        wilayah[prov].setdefault(kota, set())
        wilayah[prov][kota].add(kec)

    # Ubah set → sorted list agar JSON-serializable
    wilayah_sorted = {
        prov: {
            kota: sorted(kec_set)
            for kota, kec_set in sorted(kota_dict.items())
        }
        for prov, kota_dict in sorted(wilayah.items())
    }

    return jsonify({
        'total_properti': len(df_properti),
        'wilayah': wilayah_sorted,               # <-- struktur bertingkat baru
        'provinsi_list': sorted(wilayah_sorted.keys()),
        'kota_list': sorted(df_properti['Kota_Kab'].dropna().unique().tolist()),
        'kecamatan_list': sorted(df_properti['Kecamatan'].dropna().unique().tolist()),
        'jenis_list': sorted(df_properti['Jenis_Properti'].dropna().unique().tolist()),
        'price_range': {
            'min': int(df_properti['Price_Clean'].min()),
            'max': int(df_properti['Price_Clean'].max())
        },
        'luas_range': {
            'min': int(df_properti['Luas_bangunan'].min()),
            'max': int(df_properti['Luas_bangunan'].max())
        }
    })


@app.route('/api/rekomendasi', methods=['POST'])
def get_rekomendasi():
    """Endpoint utama: rekomendasi dengan filter Provinsi → Kota → Kecamatan"""
    data = request.json

    preferences = {
        'max_price':      data.get('max_price', 0),
        'min_luas':       data.get('min_luas', 0),
        'kamar_tidur':    data.get('kamar_tidur', 0),
        'jenis_properti': data.get('jenis_properti', ''),
        'provinsi':       data.get('provinsi', ''),
        'kota':           data.get('kota', ''),
        'kecamatan':      data.get('kecamatan', ''),
        'pekerjaan':      data.get('pekerjaan', 'Umum'),
    }

    top_k = int(data.get('top_k', 10))

    # ── Bobot AHP berdasarkan profesi ────────────────────────────
    pekerjaan_user = preferences['pekerjaan'].lower()
    if pekerjaan_user == 'buruh':
        weights = [0.45, 0.10, 0.10, 0.05, 0.30]
    elif pekerjaan_user == 'asn':
        weights = [0.25, 0.20, 0.15, 0.10, 0.30]
    elif pekerjaan_user == 'pengusaha':
        weights = [0.10, 0.35, 0.25, 0.15, 0.15]
    else:
        weights = [0.35, 0.20, 0.15, 0.10, 0.20]

    ahp_result = {
        'kategori_konsumen': preferences['pekerjaan'],
        'weights': weights,
        'consistency_ratio': 0.0,
        'is_consistent': True
    }

    # ── HARD FILTER WILAYAH (Provinsi → Kota → Kecamatan) ────────
    df_scoring = df_properti.copy()

    if preferences['provinsi'] and preferences['provinsi'].lower() not in ('', 'semua'):
        df_scoring = df_scoring[
            df_scoring['Provinsi'].str.lower() == preferences['provinsi'].lower()
        ]

    if preferences['kota'] and preferences['kota'].lower() not in ('', 'semua'):
        df_scoring = df_scoring[
            df_scoring['Kota_Kab'].str.lower() == preferences['kota'].lower()
        ]

    if preferences['kecamatan'] and preferences['kecamatan'].lower() not in ('', 'semua'):
        df_scoring = df_scoring[
            df_scoring['Kecamatan'].str.lower() == preferences['kecamatan'].lower()
        ]

    if df_scoring.empty:
        return jsonify({
            'recommendations': [],
            'ahp': ahp_result,
            'metrics': {
                'precision_at_k': 0, 'recall_at_k': 0,
                'f1_at_k': 0, 'ndcg_at_k': 0,
                'mrr': 0, 'k': top_k
            },
            'total_scored': 0,
            'pesan': f"Tidak ada properti ditemukan untuk wilayah yang dipilih."
        })

    # ── Profile Matching ─────────────────────────────────────────
    scores = profile_matching(df_scoring, preferences, weights)

    # ── Ranking ──────────────────────────────────────────────────
    scores_sorted = sorted(scores, key=lambda x: x['final_score'], reverse=True)
    top_results   = scores_sorted[:top_k]

    # ── Gabungkan dengan data properti ───────────────────────────
    recommendations = []
    for item in top_results:
        row = df_properti.loc[item['index']]
        recommendations.append({
            'rank':           len(recommendations) + 1,
            'harga':          int(row['Price_Clean']),
            'luas_bangunan':  int(row['Luas_bangunan']),
            'kamar_tidur':    int(row['Kamar_tidur_clean']),
            'jenis_properti': str(row['Jenis_Properti']),
            'kecamatan':      str(row.get('Kecamatan', '-')),
            'kota':           str(row.get('Kota_Kab', '-')),
            'provinsi':       str(row.get('Provinsi', '-')),
            'skor':           item['final_score'],
            'detail_skor':    item['scores']
        })

    # ── Metrik Evaluasi ───────────────────────────────────────────
    all_indices  = [s['index'] for s in scores_sorted]
    relevant_set = [s['index'] for s in scores_sorted if s['final_score'] >= 3.5]
    prec  = precision_at_k(all_indices, relevant_set, top_k)
    rec   = recall_at_k(all_indices, relevant_set, top_k)
    f1    = f1_at_k(prec, rec)
    ndcg  = ndcg_at_k(all_indices, relevant_set, top_k)
    mrr_val = mrr(all_indices, relevant_set)

    return jsonify({
        'recommendations': recommendations,
        'ahp': ahp_result,
        'metrics': {
            'precision_at_k': round(prec, 4),
            'recall_at_k':    round(rec, 4),
            'f1_at_k':        round(f1, 4),
            'ndcg_at_k':      round(ndcg, 4),
            'mrr':            round(mrr_val, 4),
            'k':              top_k
        },
        'total_scored': len(scores)
    })


# ===================== RUN SERVER =====================
if __name__ == '__main__':
    print(f"Dataset loaded: {len(df_properti)} properti")
    print(f"Provinsi terdeteksi: {sorted(df_properti['Provinsi'].unique().tolist())}")
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)
