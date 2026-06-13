import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI HALAMAN WEB
# ==============================================================================
st.set_page_config(page_title="DSS Peminatan Mahasiswa", page_icon="🎓", layout="wide")

st.title("🎓 Decision Support System: Analisis Pola Peminatan Mahasiswa")
st.markdown("Sistem rekomendasi mata kuliah berbasis **Algoritma Apriori** (Metodologi CRISP-DM)")

# ==============================================================================
# SIDEBAR: INPUT DATA & PARAMETER
# ==============================================================================
st.sidebar.header("⚙️ Pengaturan Sistem")

# 1. Upload File
uploaded_file = st.sidebar.file_uploader("Unggah file data_mahasiswa.xlsx", type=["xlsx"])

# 2. Parameter Algoritma
st.sidebar.subheader("Parameter Algoritma Apriori")
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.50, 0.10, 0.01, help="Seberapa sering pola muncul (0.10 = 10%)")
min_confidence = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.50, 0.05, help="Tingkat keyakinan rekomendasi (0.50 = 50%)")

# ==============================================================================
# FUNGSI PEMROSESAN DATA (CRISP-DM: Data Understanding & Preparation)
# ==============================================================================
@st.cache_data # Cache agar tidak proses ulang setiap kali klik
def proses_data(file):
    # Baca file tanpa header dulu untuk mencari baris "No"
    df_raw = pd.read_excel(file, sheet_name='Data Mahasiswa', header=None)
    header_idx = df_raw[df_raw[0].astype(str).str.strip() == 'No'].index[0]
    
    # Baca ulang dengan header yang benar
    df = pd.read_excel(file, sheet_name='Data Mahasiswa', header=header_idx)
    
    # Hapus baris TOTAL
    df = df[df['No'].astype(str).str.isdigit()].reset_index(drop=True)
    
    # Ambil kolom MK dan ubah ke Boolean
    kolom_mk = [
        'Data mining', 'Desain Grafis', 'Sistem Informasi Pendidikan', 
        'Teknologi IoT', 'Pengolahan Citra Digital', 'Pemrograman CMS', 
        'Realitas Virtual', 'Game Edukasi', 'Sistem Keamanan Jaringan'
    ]
    df_mk = df[kolom_mk].copy()
    df_mk_bool = df_mk.replace({'✓': True, '✗': False})
    
    return df, df_mk_bool

# ==============================================================================
# FUNGSI MODELING & EVALUATION (CRISP-DM: Modeling & Evaluation)
# ==============================================================================
@st.cache_data
def jalankan_apriori(df_bool, support, confidence):
    # 1. Frequent Itemsets
    frequent_itemsets = apriori(df_bool, min_support=support, use_colnames=True)
    
    # 2. Association Rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
    
    # Urutkan berdasarkan Lift tertinggi
    rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return rules_sorted

# ==============================================================================
# TAMPILAN UTAMA (MAIN UI)
# ==============================================================================
if uploaded_file is not None:
    st.success("✅ File berhasil diunggah!")
    
    # Jalankan proses
    df_asli, df_bool = proses_data(uploaded_file)
    rules_df = jalankan_apriori(df_bool, min_support, min_confidence)
    
    # TAB 1: Dashboard & Rekomendasi (Deployment)
    tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi (DSS)", "📊 Hasil Aturan Asosiasi", "📁 Data Mentah"])
    
    with tab1:
        st.header("💡 Sistem Rekomendasi Mata Kuliah")
        st.markdown("Pilih mata kuliah yang diminati/diambil mahasiswa, sistem akan memberikan rekomendasi mata kuliah pelengkap.")
        
        daftar_mk = df_bool.columns.tolist()
        pilihan_mk = st.selectbox("Pilih Mata Kuliah yang Diambil:", daftar_mk)
        
        if pilihan_mk:
            # Cari aturan di mana MK yang dipilih ada di antecedents (kiri)
            aturan_terkait = rules_df[rules_df['antecedents'].apply(lambda x: pilihan_mk in x)]
            
            if aturan_terkait.empty:
                st.warning(f"⚠️ Tidak ditemukan pola asosiasi yang kuat untuk '{pilihan_mk}' dengan parameter saat ini. Coba turunkan nilai Minimum Support/Confidence di sidebar.")
            else:
                # Ambil rekomendasi terbaik (confidence tertinggi)
                best_rule = aturan_terkait.iloc[0]
                rekomendasi = list(best_rule['consequents'])
                conf = best_rule['confidence'] * 100
                lift = best_rule['lift']
                
                st.info(f"### 🎯 Rekomendasi untuk: **{pilihan_mk}**")
                st.markdown(f"Mahasiswa sangat direkomendasikan untuk juga mengambil mata kuliah: **{', '.join(rekomendasi)}**")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tingkat Keyakinan (Confidence)", f"{conf:.1f}%")
                col2.metric("Kekuatan Hubungan (Lift)", f"{lift:.2f}")
                col3.metric("Frekuensi (Support)", f"{best_rule['support']*100:.1f}%")
                
                st.caption("*(Confidence > 50% dan Lift > 1 menandakan rekomendasi yang sangat kuat dan saling menguatkan)*")

    with tab2:
        st.header("📊 Tabel Aturan Asosiasi (Association Rules)")
        st.markdown("Berikut adalah seluruh pola hubungan antar mata kuliah yang ditemukan oleh algoritma.")
        
        # Format kolom agar rapi di web (ubah frozenset jadi string)
        df_tampilan = rules_df.copy()
        df_tampilan['antecedents'] = df_tampilan['antecedents'].apply(lambda x: ", ".join(list(x)))
        df_tampilan['consequents'] = df_tampilan['consequents'].apply(lambda x: ", ".join(list(x)))
        
        st.dataframe(
            df_tampilan[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
            use_container_width=True,
            hide_index=True
        )
        
        # Tombol Download
        csv = df_tampilan.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Aturan (CSV)",
            data=csv,
            file_name='hasil_aturan_asosiasi.csv',
            mime='text/csv',
        )

    with tab3:
        st.header("📁 Preview Data Mahasiswa")
        st.dataframe(df_asli.head(10), use_container_width=True)
        st.caption(f"Total data mahasiswa yang valid: {len(df_asli)} baris")

else:
    st.info("👈 Silakan unggah file `data_mahasiswa.xlsx` di sidebar sebelah kiri untuk memulai analisis.")