import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI HALAMAN WEB
# ==============================================================================
st.set_page_config(
    page_title="DSS Peminatan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang lebih baik
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER & BUSINESS UNDERSTANDING
# ==============================================================================
st.markdown('<p class="main-header">🎓 DECISION SUPPORT SYSTEM: ANALISIS POLA PEMINATAN MAHASISWA</p>', unsafe_allow_html=True)
st.markdown("### Berbasis Algoritma Apriori dengan Metodologi CRISP-DM")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📋 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["🏠 Home & Business Understanding", 
     "📊 Data Understanding & Exploration", 
     "⚙️ Data Preparation",
     "🔍 Modeling (Apriori)",
     "📈 Evaluation & Analysis",
     "🚀 Deployment & Rekomendasi"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Algoritma")
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.50, 0.10, 0.01, 
                                 help="Seberapa sering pola muncul (0.10 = 10%)")
min_confidence = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.50, 0.05,
                                    help="Tingkat keyakinan rekomendasi (0.50 = 50%)")

# Upload File
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah file Excel", type=["xlsx"])

# ==============================================================================
# FUNGSI PEMROSESAN DATA
# ==============================================================================
@st.cache_data
def proses_data(file):
    """Fungsi untuk membaca dan memproses data dari Excel"""
    try:
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
        
        return df, df_mk_bool, kolom_mk
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None, None, None

@st.cache_data
def jalankan_apriori(df_bool, support, confidence):
    """Fungsi untuk menjalankan algoritma Apriori"""
    try:
        # 1. Frequent Itemsets
        frequent_itemsets = apriori(df_bool, min_support=support, use_colnames=True)
        
        # 2. Association Rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        
        # Urutkan berdasarkan Lift tertinggi
        rules_sorted = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
        return rules_sorted, frequent_itemsets
    except Exception as e:
        st.error(f"Error menjalankan Apriori: {e}")
        return None, None

# ==============================================================================
# MAIN CONTENT
# ==============================================================================

if uploaded_file is not None:
    df_asli, df_bool, kolom_mk = proses_data(uploaded_file)
    
    if df_asli is not None:
        rules_df, frequent_itemsets = jalankan_apriori(df_bool, min_support, min_confidence)
        
        # ======================================================================
        # MENU 1: HOME & BUSINESS UNDERSTANDING
        # ======================================================================
        if menu == "🏠 Home & Business Understanding":
            st.markdown('<p class="sub-header">📋 BUSINESS UNDERSTANDING</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Tujuan Bisnis")
                st.info("""
                1. **Untuk Mahasiswa**: Memberikan rekomendasi mata kuliah yang saling melengkapi
                2. **Untuk Program Studi**: Memahami pola peminatan untuk perencanaan kurikulum
                3. **Untuk Dosen Wali**: Memiliki dasar data dalam memberikan konsultasi akademik
                """)
                
                st.markdown("#### 👥 Stakeholder")
                st.success("""
                - **Mahasiswa Baru**: Mendapat rekomendasi MK yang tepat
                - **Dosen Wali**: Dasar data untuk konsultasi
                - **Kaprodi**: Insight untuk perencanaan kurikulum
                - **Bagian Akademik**: Optimasi penjadwalan
                """)
            
            with col2:
                st.markdown("#### ✅ Kriteria Kesuksesan")
                st.warning(f"""
                - Menemukan minimal 10 aturan asosiasi dengan Lift > 1.2
                - Confidence rekomendasi minimal 50%
                - Support minimal {min_support*100:.0f}%
                - Sistem dapat memberikan rekomendasi cepat
                """)
            
            st.markdown("---")
            st.markdown("#### 🔄 Metodologi: CRISP-DM")
            st.markdown("""
            Proyek ini menggunakan **CRISP-DM (Cross-Industry Standard Process for Data Mining)** yang terdiri dari 6 tahap:
            
            1. **Business Understanding** - Memahami tujuan dan kebutuhan
            2. **Data Understanding** - Mengumpulkan dan mengeksplorasi data
            3. **Data Preparation** - Membersihkan dan mempersiapkan data
            4. **Modeling** - Membangun model dengan algoritma Apriori
            5. **Evaluation** - Mengevaluasi hasil model
            6. **Deployment** - Implementasi sistem rekomendasi
            
            **Parameter Saat Ini:**
            - Minimum Support: **{}** ({}%)
            - Minimum Confidence: **{}** ({}%)
            """.format(min_support, min_support*100, min_confidence, min_confidence*100))
            
            st.markdown("---")
            st.success("✅ Data berhasil diunggah! Total **{} mahasiswa** dengan **{} mata kuliah**".format(
                len(df_asli), len(kolom_mk)))
        
        # ======================================================================
        # MENU 2: DATA UNDERSTANDING & EXPLORATION
        # ======================================================================
        elif menu == "📊 Data Understanding & Exploration":
            st.markdown('<p class="sub-header">📊 DATA UNDERSTANDING & EXPLORATION</p>', unsafe_allow_html=True)
            
            # Statistik Deskriptif
            st.markdown("#### 📈 Statistik Deskriptif")
            df_asli['Jumlah_MK'] = df_bool.sum(axis=1)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Mahasiswa", len(df_asli))
            col2.metric("Total Mata Kuliah", len(kolom_mk))
            col3.metric("Rata-rata MK/Mahasiswa", "{:.2f}".format(df_asli['Jumlah_MK'].mean()))
            col4.metric("Median MK/Mahasiswa", "{:.0f}".format(df_asli['Jumlah_MK'].median()))
            
            # Popularitas Mata Kuliah
            st.markdown("---")
            st.markdown("#### 📊 Popularitas Mata Kuliah")
            popularitas = df_bool.sum().sort_values(ascending=False)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Eksplorasi Data Pemilihan Mata Kuliah', fontsize=14, fontweight='bold')
            
            # Plot 1: Histogram Jumlah MK
            axes[0, 0].hist(df_asli['Jumlah_MK'], bins=range(1, 10), 
                           edgecolor='black', color='steelblue', alpha=0.7)
            axes[0, 0].set_title('Distribusi Jumlah MK per Mahasiswa')
            axes[0, 0].set_xlabel('Jumlah Mata Kuliah')
            axes[0, 0].set_ylabel('Frekuensi')
            axes[0, 0].axvline(df_asli['Jumlah_MK'].mean(), color='red', 
                              linestyle='--', label='Mean: {:.1f}'.format(df_asli["Jumlah_MK"].mean()))
            axes[0, 0].legend()
            
            # Plot 2: Bar Chart Popularitas
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(popularitas)))
            axes[0, 1].barh(popularitas.index, popularitas.values, color=colors)
            axes[0, 1].set_title('Popularitas Mata Kuliah')
            axes[0, 1].set_xlabel('Jumlah Mahasiswa')
            
            # Plot 3: Pie Chart (diperbaiki)
            explode = [0.05] * len(popularitas)
            # Pastikan data numerik
            pie_values = popularitas.values.astype(float)
            axes[1, 0].pie(pie_values, labels=popularitas.index, 
                          autopct='%1.1f%%', startangle=90, explode=explode)
            axes[1, 0].set_title('Persentase Pemilihan MK')
            
            # Plot 4: Box Plot
            axes[1, 1].boxplot(df_asli['Jumlah_MK'], vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightblue'))
            axes[1, 1].set_title('Box Plot Jumlah MK')
            axes[1, 1].set_ylabel('Jumlah Mata Kuliah')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabel Popularitas - DIPERBAIKI
            st.markdown("#### 📋 Detail Popularitas")
            
            # Fix: Convert to proper format before rounding
            persentase_values = [round((val / len(df_asli)) * 100, 1) for val in popularitas.values]
            
            df_pop = pd.DataFrame({
                'Mata Kuliah': popularitas.index,
                'Jumlah Mahasiswa': popularitas.values,
                'Persentase': persentase_values
            })
            st.dataframe(df_pop, use_container_width=True)
            
            # Heatmap Korelasi
            st.markdown("---")
            st.markdown("#### 🔥 Heatmap Korelasi Antar Mata Kuliah")
            correlation_matrix = df_bool.corr()
            
            fig2, ax = plt.subplots(figsize=(12, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdYlGn', center=0, ax=ax, annot_kws={"size": 8})
            plt.title('Heatmap Korelasi Antar Mata Kuliah', fontweight='bold')
            st.pyplot(fig2)
        
        # ======================================================================
        # MENU 3: DATA PREPARATION
        # ======================================================================
        elif menu == "⚙️ Data Preparation":
            st.markdown('<p class="sub-header">⚙️ DATA PREPARATION</p>', unsafe_allow_html=True)
            
            st.markdown("#### Proses Persiapan Data:")
            st.info("""
            1. **Membaca Data**: Mengambil data dari sheet 'Data Mahasiswa'
            2. **Deteksi Header**: Mencari baris yang mengandung kolom 'No'
            3. **Cleaning**: Menghapus baris 'TOTAL' dan data non-numerik
            4. **Transformasi**: Mengubah simbol ✓ menjadi True dan ✗ menjadi False
            5. **Validasi**: Memastikan data siap untuk algoritma Apriori
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Data Asli (5 Baris Pertama)")
                st.dataframe(df_asli.head(), use_container_width=True)
            
            with col2:
                st.markdown("#### Data Boolean (5 Baris Pertama)")
                st.dataframe(df_bool.head(), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### ✅ Konversi Data")
            st.write("**Simbol yang digunakan:**")
            st.write("✓ = True (Mahasiswa mengambil MK)")
            st.write("✗ = False (Mahasiswa tidak mengambil MK)")
            
            st.success("✅ Data preparation selesai! {} baris data valid siap diproses.".format(len(df_asli)))
        
        # ======================================================================
        # MENU 4: MODELING
        # ======================================================================
        elif menu == "🔍 Modeling (Apriori)":
            st.markdown('<p class="sub-header">🔍 MODELING - ALGORITMA APRIORI</p>', unsafe_allow_html=True)
            
            st.markdown("#### 📚 Penjelasan Algoritma Apriori")
            st.markdown("""
            **Apriori** adalah algoritma untuk mencari pola asosiasi (association rules) dalam dataset.
            
            **3 Metrik Utama:**
            1. **Support**: Seberapa sering kombinasi item muncul bersama
            2. **Confidence**: Tingkat keyakinan bahwa jika A maka B
            3. **Lift**: Kekuatan hubungan antara A dan B (Lift > 1 = hubungan positif)
            
            **Rumus:**
            - Support(A→B) = P(A∩B)
            - Confidence(A→B) = P(B|A) = P(A∩B) / P(A)
            - Lift(A→B) = Confidence(A→B) / Support(B)
            """)
            
            if rules_df is not None and len(rules_df) > 0:
                st.markdown("---")
                st.markdown("#### 📊 Hasil Modeling")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Frequent Itemsets", len(frequent_itemsets))
                col2.metric("Total Aturan Asosiasi", len(rules_df))
                col3.metric("Aturan dengan Lift > 1.2", len(rules_df[rules_df['lift'] > 1.2]))
                
                st.markdown("#### Top 10 Aturan Terbaik (Berdasarkan Lift)")
                df_top10 = rules_df.head(10).copy()
                df_top10['antecedents'] = df_top10['antecedents'].apply(lambda x: ", ".join(list(x)))
                df_top10['consequents'] = df_top10['consequents'].apply(lambda x: ", ".join(list(x)))
                
                st.dataframe(
                    df_top10[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Tidak ada aturan yang ditemukan. Coba turunkan nilai Support atau Confidence.")
        
        # ======================================================================
        # MENU 5: EVALUATION
        # ======================================================================
        elif menu == "📈 Evaluation & Analysis":
            st.markdown('<p class="sub-header">📈 EVALUATION & ANALYSIS</p>', unsafe_allow_html=True)
            
            if rules_df is not None and len(rules_df) > 0:
                # Distribusi Metrik
                st.markdown("#### 📊 Distribusi Metrik")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].hist(rules_df['support'], bins=20, color='steelblue', edgecolor='black')
                axes[0].set_title('Distribusi Support')
                axes[0].set_xlabel('Support')
                axes[0].set_ylabel('Frekuensi')
                
                axes[1].hist(rules_df['confidence'], bins=20, color='green', edgecolor='black')
                axes[1].set_title('Distribusi Confidence')
                axes[1].set_xlabel('Confidence')
                axes[1].set_ylabel('Frekuensi')
                
                axes[2].hist(rules_df['lift'], bins=20, color='orange', edgecolor='black')
                axes[2].set_title('Distribusi Lift')
                axes[2].set_xlabel('Lift')
                axes[2].set_ylabel('Frekuensi')
                axes[2].axvline(x=1.0, color='red', linestyle='--', label='Lift = 1')
                axes[2].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Analisis Mendalam
                st.markdown("---")
                st.markdown("#### 💡 Analisis Mendalam Aturan Terbaik")
                
                for i, row in rules_df.head(5).iterrows():
                    with st.expander("ATURAN #{}: Lift = {:.3f}".format(i+1, row['lift'])):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**JIKA (Antecedents):**")
                            st.write(", ".join(list(row['antecedents'])))
                        with col2:
                            st.markdown("**MAKA (Consequents):**")
                            st.write(", ".join(list(row['consequents'])))
                        
                        st.markdown("---")
                        st.markdown("**Metrik:**")
                        st.write("- **Support**: {:.3f} ({:.1f}%)".format(row['support'], row['support']*100))
                        st.write("- **Confidence**: {:.3f} ({:.1f}%)".format(row['confidence'], row['confidence']*100))
                        st.write("- **Lift**: {:.3f}".format(row['lift']))
                        
                        # Interpretasi
                        st.markdown("**Interpretasi:**")
                        if row['lift'] > 1.3:
                            st.success("🟢 Hubungan SANGAT KUAT - Rekomendasi sangat direkomendasikan")
                        elif row['lift'] > 1.1:
                            st.info("🟡 Hubungan KUAT - Rekomendasi dapat diandalkan")
                        else:
                            st.warning("🟠 Hubungan SEDANG - Perlu pertimbangan tambahan")
                
                # Pola Khusus
                st.markdown("---")
                st.markdown("#### 🔍 Pola Khusus yang Menarik")
                
                # MK yang paling sering jadi antecedent
                antecedent_count = {}
                for rule in rules_df.itertuples():
                    for mk in rule.antecedents:
                        antecedent_count[mk] = antecedent_count.get(mk, 0) + 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**MK Paling Sering Jadi Pemicu:**")
                    sorted_ant = sorted(antecedent_count.items(), key=lambda x: x[1], reverse=True)
                    for mk, count in sorted_ant[:5]:
                        st.write("- {}: {} aturan".format(mk, count))
                
                with col2:
                    # MK yang paling sering direkomendasikan
                    consequent_count = {}
                    for rule in rules_df.itertuples():
                        for mk in rule.consequents:
                            consequent_count[mk] = consequent_count.get(mk, 0) + 1
                    
                    st.markdown("**MK Paling Sering Direkomendasikan:**")
                    sorted_con = sorted(consequent_count.items(), key=lambda x: x[1], reverse=True)
                    for mk, count in sorted_con[:5]:
                        st.write("- {}: {} aturan".format(mk, count))
            else:
                st.warning("⚠️ Tidak ada data untuk dievaluasi.")
        
        # ======================================================================
        # MENU 6: DEPLOYMENT
        # ======================================================================
        elif menu == "🚀 Deployment & Rekomendasi":
            st.markdown('<p class="sub-header">🚀 DEPLOYMENT & SISTEM REKOMENDASI</p>', unsafe_allow_html=True)
            
            st.markdown("#### 💡 Sistem Rekomendasi Interaktif")
            st.markdown("Pilih mata kuliah yang diminati, sistem akan memberikan rekomendasi:")
            
            if rules_df is not None and len(rules_df) > 0:
                pilihan_mk = st.selectbox("Pilih Mata Kuliah yang Diambil:", kolom_mk)
                
                if pilihan_mk:
                    # Cari aturan yang relevan
                    aturan_terkait = rules_df[rules_df['antecedents'].apply(lambda x: pilihan_mk in x)]
                    
                    if aturan_terkait.empty:
                        st.warning("⚠️ Tidak ditemukan pola asosiasi untuk '{}'".format(pilihan_mk))
                    else:
                        st.markdown("### 🎯 Rekomendasi untuk: **{}**".format(pilihan_mk))
                        
                        # Kelompokkan rekomendasi yang unik
                        rekomendasi_unik = {}
                        for idx, row in aturan_terkait.iterrows():
                            consequents = tuple(sorted(row['consequents']))
                            if consequents not in rekomendasi_unik:
                                rekomendasi_unik[consequents] = {
                                    'confidence': row['confidence'],
                                    'lift': row['lift'],
                                    'support': row['support']
                                }
                            else:
                                # Ambil yang confidence tertinggi
                                if row['confidence'] > rekomendasi_unik[consequents]['confidence']:
                                    rekomendasi_unik[consequents] = {
                                        'confidence': row['confidence'],
                                        'lift': row['lift'],
                                        'support': row['support']
                                    }
                        
                        # Tampilkan rekomendasi unik
                        for i, (consequents, metrics) in enumerate(sorted(
                            rekomendasi_unik.items(), 
                            key=lambda x: x[1]['confidence'], 
                            reverse=True
                        ), 1):
                            with st.container():
                                st.markdown("##### Rekomendasi #{}".format(i))
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown("**Ambil juga:** {}".format(", ".join(consequents)))
                                with col2:
                                    st.metric("Confidence", "{:.1f}%".format(metrics['confidence']*100))
                                
                                col3, col4 = st.columns(2)
                                col3.write("Lift: {:.3f}".format(metrics['lift']))
                                col4.write("Support: {:.1f}%".format(metrics['support']*100))
                                st.markdown("---")
                
                # Download Hasil - DIPERBAIKI DENGAN EXCEL YANG RAPI
                st.markdown("---")
                st.markdown("#### 📥 Download Hasil Analisis")
                
                # Siapkan data untuk Excel yang lebih rapi
                df_export = rules_df.copy()
                df_export['antecedents'] = df_export['antecedents'].apply(lambda x: ", ".join(list(x)))
                df_export['consequents'] = df_export['consequents'].apply(lambda x: ", ".join(list(x)))
                
                # Format kolom dengan nama yang lebih baik
                df_export_formatted = df_export.rename(columns={
                    'antecedents': 'Jika Mahasiswa Mengambil',
                    'consequents': 'Maka Direkomendasikan',
                    'support': 'Support',
                    'confidence': 'Confidence',
                    'lift': 'Lift'
                })
                
                # Tambahkan kolom persentase yang lebih mudah dibaca
                df_export_formatted['Support (%)'] = (df_export_formatted['Support'] * 100).round(2)
                df_export_formatted['Confidence (%)'] = (df_export_formatted['Confidence'] * 100).round(2)
                df_export_formatted['Lift'] = df_export_formatted['Lift'].round(3)
                
                # Download CSV
                csv = df_export_formatted.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    label="📥 Download Hasil Aturan (CSV)",
                    data=csv,
                    file_name='hasil_aturan_asosiasi.csv',
                    mime='text/csv',
                )
                
                # Download Excel dengan formatting yang lebih baik
                from io import BytesIO
                output = BytesIO()
                
                # Gunakan ExcelWriter untuk formatting yang lebih baik
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_export_formatted.to_excel(writer, sheet_name='Association Rules', index=False)
                    
                    # Akses workbook dan worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Association Rules']
                    
                    # Format header
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#1f77b4',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    # Format angka
                    number_format = workbook.add_format({'num_format': '0.00'})
                    percent_format = workbook.add_format({'num_format': '0.00%'})
                    
                    # Terapkan format
                    for col_num, value in enumerate(df_export_formatted.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Set column width
                    worksheet.set_column('A:A', 35)  # Jika Mahasiswa Mengambil
                    worksheet.set_column('B:B', 35)  # Maka Direkomendasikan
                    worksheet.set_column('C:C', 12)  # Support
                    worksheet.set_column('D:D', 15)  # Support (%)
                    worksheet.set_column('E:E', 12)  # Confidence
                    worksheet.set_column('F:F', 15)  # Confidence (%)
                    worksheet.set_column('G:G', 10)  # Lift
                
                output.seek(0)
                
                st.download_button(
                    label="📥 Download Hasil Aturan (Excel - Format Rapi)",
                    data=output,
                    file_name='hasil_aturan_asosiasi.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else:
                st.warning("⚠️ Tidak ada aturan untuk ditampilkan.")
    
else:
    st.info("👈 Silakan unggah file `data_mahasiswa.xlsx` di sidebar untuk memulai analisis.")
    
    st.markdown("""
    ---
    ### 📖 Cara Menggunakan:
    1. Upload file Excel di sidebar
    2. Atur parameter Support dan Confidence
    3. Navigasi melalui menu di sidebar
    4. Lihat hasil analisis dan rekomendasi
    
    **Format Data yang Diharapkan:**
    - Sheet name: 'Data Mahasiswa'
    - Kolom: No, Nama, NIM, dan nama-nama mata kuliah
    - Simbol: ✓ (mengambil) dan ✗ (tidak mengambil)
    """)