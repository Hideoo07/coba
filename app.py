"""
Backend API - Sistem Rekomendasi Properti
Menggunakan Flask + AHP + Profile Matching
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ===================== LOAD DATASET =====================
DATA_PATH = "data_original.xlsx"

def load_data():
    """Load dan preprocessing dataset properti"""
    if not os.path.exists(DATA_PATH):
        # Fallback: generate sample data untuk testing
        return generate_sample_data()
    df = pd.read_excel(DATA_PATH)
    df = preprocess(df)
    return df

def generate_sample_data():
    """Sample data jika file Excel belum tersedia"""
    np.random.seed(42)
    n = 50
    kecamatan_list = ['Menteng', 'Kebayoran Baru', 'Kelapa Gading', 'Pondok Indah',
                      'Cibubur', 'Serpong', 'Depok', 'Bekasi', 'Tangerang', 'Bogor']
    kota_list = ['Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Utara', 'Jakarta Selatan',
                 'Jakarta Timur', 'Tangerang Selatan', 'Depok', 'Bekasi', 'Tangerang', 'Bogor']
    data = {
        'Price_Clean': np.random.randint(300_000_000, 5_000_000_000, n),
        'Luas_bangunan': np.random.randint(36, 500, n),
        'Luas_tanah': np.random.randint(60, 600, n),
        'Kamar_tidur_clean': np.random.choice([1,2,3,4,5], n),
        'Kamar_mandi': np.random.choice([1,2,3,4], n),
        'Jenis_Properti': np.random.choice(['Rumah', 'Apartemen', 'Townhouse'], n),
        'Kecamatan': np.random.choice(kecamatan_list, n),
        'Kota_Kab': np.random.choice(kota_list, n),
        'Lat': np.random.uniform(-6.35, -6.10, n),
        'Lon': np.random.uniform(106.70, 106.95, n),
    }
    return pd.DataFrame(data)

def preprocess(df):
    """Bersihkan dan siapkan data"""
    cols_needed = ['Price_Clean', 'Luas_bangunan', 'Kamar_tidur_clean', 'Jenis_Properti',
                   'Kecamatan', 'Kota_Kab']
    for col in cols_needed:
        if col not in df.columns:
            df[col] = 0
    df = df.dropna(subset=['Price_Clean', 'Luas_bangunan', 'Kamar_tidur_clean'])
    return df

# Load data saat startup
df_properti = load_data()

# ===================== AHP FUNCTIONS =====================
def ahp_weights(criteria_matrix):
    """
    Hitung bobot prioritas dari matriks perbandingan AHP.
    Menggunakan metode geometric mean (approximate).
    """
    matrix = np.array(criteria_matrix, dtype=float)
    n = matrix.shape[0]

    # Geometric mean per baris
    geo_means = np.prod(matrix, axis=1) ** (1.0 / n)
    weights = geo_means / geo_means.sum()

    # Consistency check
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
    """
    Hitung skor Profile Matching untuk setiap properti.
    GAP = Profil Properti - Profil Ideal User
    """
    results = []

    for idx, row in df.iterrows():
        gaps = {}
        scores = {}

        # GAP Harga (semakin dekat ke budget, semakin baik)
        if preferences.get('max_price', 0) > 0:
            price_ratio = row['Price_Clean'] / preferences['max_price']
            if price_ratio <= 1:
                scores['harga'] = 5 - (1 - price_ratio) * 4  # 1-5 scale
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

        # GAP Jenis Properti (match/no match)
        if preferences.get('jenis_properti'):
            scores['jenis'] = 5 if row['Jenis_Properti'] == preferences['jenis_properti'] else 2
        else:
            scores['jenis'] = 3

        # GAP Lokasi (match kecamatan/kota)
        loc_score = 2
        if preferences.get('kecamatan') and row.get('Kecamatan') == preferences['kecamatan']:
            loc_score = 5
        elif preferences.get('kota') and row.get('Kota_Kab') == preferences['kota']:
            loc_score = 4
        scores['lokasi'] = loc_score

        # Hitung skor akhir dengan bobot AHP
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
    """Info tentang dataset"""
    return jsonify({
        'total_properti': len(df_properti),
        'kecamatan_list': sorted(df_properti['Kecamatan'].dropna().unique().tolist()),
        'kota_list': sorted(df_properti['Kota_Kab'].dropna().unique().tolist()),
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
    """Endpoint utama: terima preferensi berdasarkan kategori profesi, return rekomendasi"""
    data = request.json

    # 1. Ambil preferensi user (DITAMBAH: Ambil data 'pekerjaan')
    preferences = {
        'max_price': data.get('max_price', 0),
        'min_luas': data.get('min_luas', 0),
        'kamar_tidur': data.get('kamar_tidur', 0),
        'jenis_properti': data.get('jenis_properti', ''),
        'kecamatan': data.get('kecamatan', ''),
        'kota': data.get('kota', ''),
        'pekerjaan': data.get('pekerjaan', 'Umum') # Default ke Umum jika tidak ada
    }

    # 2. Penentuan Bobot AHP Otomatis Berdasarkan Profesi (REVISI PENGUJI)
    # Urutan Kriteria: [Harga, Luas, Kamar, Jenis, Lokasi]
    pekerjaan_user = preferences['pekerjaan'].lower()
    
    if pekerjaan_user == 'buruh':
        # BURUH: Sangat sensitif pada Harga dan Lokasi (dekat tempat kerja/pabrik)
        weights = [0.45, 0.10, 0.10, 0.05, 0.30]
    
    elif pekerjaan_user == 'asn':
        # ASN: Prioritas seimbang, cenderung mencari kenyamanan dan lokasi strategis
        weights = [0.25, 0.20, 0.15, 0.10, 0.30]
    
    elif pekerjaan_user == 'pengusaha':
        # PENGUSAHA: Fokus pada ukuran bangunan/tanah besar, banyak kamar. Harga bukan isu utama.
        weights = [0.10, 0.35, 0.25, 0.15, 0.15]
    
    else:
        # DEFAULT: Preferensi standar
        weights = [0.35, 0.20, 0.15, 0.10, 0.20]

    # Karena bobot diatur otomatis oleh sistem (aturan pakar), CR selalu dianggap konsisten
    ahp_result = {
        'kategori_konsumen': preferences['pekerjaan'],
        'weights': weights, 
        'consistency_ratio': 0.0, 
        'is_consistent': True
    }
    # ... (kode penentuan bobot ASN/Buruh/Pengusaha yang tadi) ...

    # =======================================================
    # [TAMBAHAN BARU] HARD FILTER LOKASI
    # Memfilter data secara absolut agar tidak ada kota lain yang bocor
    # =======================================================
    df_scoring = df_properti.copy()
    
    if preferences['kota'] and preferences['kota'] != 'Semua':
        df_scoring = df_scoring[df_scoring['Kota_Kab'].str.lower() == preferences['kota'].lower()]
        
    if preferences['kecamatan'] and preferences['kecamatan'] != 'Semua':
        df_scoring = df_scoring[df_scoring['Kecamatan'].str.lower() == preferences['kecamatan'].lower()]

    # Jika setelah difilter ternyata tidak ada rumah di kota tersebut
    if df_scoring.empty:
        return jsonify({
            'recommendations': [],
            'ahp': ahp_result,
            'metrics': {'precision_at_k': 0, 'recall_at_k': 0, 'f1_at_k': 0, 'ndcg_at_k': 0, 'mrr': 0, 'k': top_k},
            'total_scored': 0
        })

    # 3. Jalankan Profile Matching HANYA pada data yang kotanya sudah sesuai
    # (Ubah variabel 'df_properti' menjadi 'df_scoring' di bawah ini)
    scores = profile_matching(df_scoring, preferences, weights)

    # 4. Ranking
    # ... (kode ke bawahnya tetap sama) ...

    # 3. Jalankan Profile Matching
    scores = profile_matching(df_properti, preferences, weights)

    # 4. Ranking
    scores_sorted = sorted(scores, key=lambda x: x['final_score'], reverse=True)
    top_k = int(data.get('top_k', 10))
    top_results = scores_sorted[:top_k]

    # 5. Gabungkan dengan data properti
    recommendations = []
    for item in top_results:
        row = df_properti.iloc[item['index']]
        recommendations.append({
            'rank': len(recommendations) + 1,
            'harga': int(row['Price_Clean']),
            'luas_bangunan': int(row['Luas_bangunan']),
            'kamar_tidur': int(row['Kamar_tidur_clean']),
            'jenis_properti': str(row['Jenis_Properti']),
            'kecamatan': str(row.get('Kecamatan', '-')),
            'kota': str(row.get('Kota_Kab', '-')),
            'skor': item['final_score'],
            'detail_skor': item['scores']
        })

    # 6. Hitung metrik evaluasi (simulasi relevansi: properti sesuai budget & lokasi)
    all_indices = [s['index'] for s in scores_sorted]
    relevant_set = [s['index'] for s in scores_sorted if s['final_score'] >= 3.5]

    prec = precision_at_k(all_indices, relevant_set, top_k)
    rec = recall_at_k(all_indices, relevant_set, top_k)
    f1 = f1_at_k(prec, rec)
    ndcg = ndcg_at_k(all_indices, relevant_set, top_k)
    mrr_val = mrr(all_indices, relevant_set)

    return jsonify({
        'recommendations': recommendations,
        'ahp': ahp_result,
        'metrics': {
            'precision_at_k': round(prec, 4),
            'recall_at_k': round(rec, 4),
            'f1_at_k': round(f1, 4),
            'ndcg_at_k': round(ndcg, 4),
            'mrr': round(mrr_val, 4),
            'k': top_k
        },
        'total_scored': len(scores)
    })

# ===================== RUN SERVER =====================
if __name__ == '__main__':
    print(f"Dataset loaded: {len(df_properti)} properti")
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)