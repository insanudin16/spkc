import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Random Index (RI) values for different matrix sizes (used in AHP)
RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Function to generate dummy data
def generate_dummy_data(num_data):
    data = {
        'Alternatif': [f'Lahan {i+1}' for i in range(num_data)],
        'C1 (Luas Lahan)': np.random.randint(40, 100, size=num_data),
        'C2 (Ketersediaan Air)': np.random.randint(10000, 20000, size=num_data),
        'C3 (Kedekatan Infrastruktur)': np.random.randint(5, 20, size=num_data),
        'C4 (Biaya Perolehan)': np.random.randint(80, 150, size=num_data)
    }
    df = pd.DataFrame(data)
    return df

# AHP Functions
def normalize_matrix(matrix):
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums
    return normalized_matrix

def calculate_weights(normalized_matrix):
    avg_weights = normalized_matrix.mean(axis=1)
    return avg_weights

def calculate_cr(matrix, weights):
    n = matrix.shape[0]
    weighted_sum = np.dot(matrix, weights)
    lamda_max = weighted_sum.mean() / weights.mean()
    CI = (lamda_max - n) / (n - 1)
    RI = RI_dict[n] if n in RI_dict else 1.49  # Default to RI for n = 10
    CR = CI / RI
    return CI, CR

def ahp_method(df, pairwise_matrix):
    normalized_matrix = normalize_matrix(pairwise_matrix)
    weights = calculate_weights(normalized_matrix)
    CI, CR = calculate_cr(pairwise_matrix, weights)

    if CR > 0.1:
        st.warning(f'Matriks Perbandingan tidak konsisten (CR = {CR:.4f}). Silakan periksa kembali preferensi Anda.')
    
    df['Skor AHP'] = df.iloc[:, 1:].dot(weights)
    return df, weights, CI, CR

# WP, SAW, and TOPSIS Functions
def wp_method(df, weights):
    criteria_cols = df.columns[1:-1]  # Asumsikan 'Alternatif' adalah kolom pertama dan 'Skor AHP' adalah kolom terakhir
    
    if len(weights) != len(criteria_cols):
        st.error(f"Jumlah bobot ({len(weights)}) tidak sesuai dengan jumlah kriteria ({len(criteria_cols)})")
        return df

    df['Skor WP'] = np.prod(df[criteria_cols].values ** weights, axis=1)
    return df

def saw_method(df, weights):
    criteria_cols = df.columns[1:-1]
    
    if len(weights) != len(criteria_cols):
        st.error(f"Jumlah bobot ({len(weights)}) tidak sesuai dengan jumlah kriteria ({len(criteria_cols)})")
        return df

    df_norm = df[criteria_cols] / df[criteria_cols].max()
    df['Skor SAW'] = np.dot(df_norm.values, weights)
    return df

def topsis_method(df, weights):
    criteria_cols = df.columns[1:-1]
    
    if len(weights) != len(criteria_cols):
        st.error(f"Jumlah bobot ({len(weights)}) tidak sesuai dengan jumlah kriteria ({len(criteria_cols)})")
        return df

    df_norm = df[criteria_cols] / np.sqrt((df[criteria_cols]**2).sum())
    
    weighted_norm = df_norm * weights
    ideal_best = weighted_norm.max()
    ideal_worst = weighted_norm.min()

    distance_best = np.sqrt(((weighted_norm - ideal_best)**2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_norm - ideal_worst)**2).sum(axis=1))
    
    df['Skor TOPSIS'] = distance_worst / (distance_worst + distance_best)
    return df

# Streamlit App Layout
st.sidebar.title('Navigasi')
selection = st.sidebar.selectbox('Pilih Halaman', ['Home', 'Generate Data Dummy', 'Input/Edit Data Alternatif', 'Input Matriks Perbandingan Berpasangan (AHP)', 'Perhitungan dan Komparasi Algoritma', 'Hasil Terbaik'])

if 'data_dummy' not in st.session_state:
    st.session_state['data_dummy'] = pd.DataFrame()

if selection == 'Home':
    st.title('Pemilihan Lahan di IKN Menggunakan Metode WP, SAW, AHP, dan TOPSIS')
    st.write("""
    Aplikasi ini menggunakan berbagai metode untuk memilih lahan terbaik berdasarkan kriteria seperti Luas Lahan, Ketersediaan Air, Kedekatan Infrastruktur, dan Biaya Perolehan. Anda dapat meng-generate data dummy, memasukkan bobot, melakukan perhitungan dengan berbagai metode, serta membandingkan hasil dari berbagai algoritma.
    """)

elif selection == 'Generate Data Dummy':
    st.title('Generate Dummy Data Alternatif Lahan')
    
    num_data = st.number_input('Jumlah data yang ingin di-generate', min_value=1, max_value=1000, value=10)

    if st.button('Generate Data'):
        df_dummy = generate_dummy_data(num_data)
        st.session_state['data_dummy'] = df_dummy
        st.subheader('Data Alternatif Lahan yang Dihasilkan')
        st.dataframe(df_dummy)
        csv = df_dummy.to_csv(index=False)
        st.download_button(label="Download Data sebagai CSV", data=csv, file_name='data_alternatif_lahan.csv', mime='text/csv')
        st.success(f'{num_data} baris data berhasil di-generate!')

elif selection == 'Input Matriks Perbandingan Berpasangan (AHP)':
    st.title('Input Matriks Perbandingan Berpasangan untuk AHP')

    if not st.session_state['data_dummy'].empty:
        st.subheader('Data Alternatif Lahan')
        st.dataframe(st.session_state['data_dummy'])

        st.subheader('Matriks Perbandingan Berpasangan (Isi Nilai di Bawah)')
        num_criteria = 4
        pairwise_matrix = np.ones((num_criteria, num_criteria))

        criteria_labels = ['C1 (Luas Lahan)', 'C2 (Ketersediaan Air)', 'C3 (Kedekatan Infrastruktur)', 'C4 (Biaya Perolehan)']

        for i in range(num_criteria):
            for j in range(i+1, num_criteria):
                value = st.number_input(f'Perbandingan {criteria_labels[i]} vs {criteria_labels[j]}', min_value=0.1, max_value=10.0, value=1.0)
                pairwise_matrix[i, j] = value
                pairwise_matrix[j, i] = 1 / value

        st.subheader('Matriks Perbandingan yang Dimasukkan')
        st.dataframe(pd.DataFrame(pairwise_matrix, index=criteria_labels, columns=criteria_labels))

        if st.button('Hitung AHP'):
            df_dummy = st.session_state['data_dummy']
            df_ahp, weights, CI, CR = ahp_method(df_dummy.copy(), pairwise_matrix)
            st.session_state['df_ahp'] = df_ahp # sampe siniiiiiiiiiiiiiii
            st.session_state['ahp_weights'] = weights

            st.subheader('Bobot Kriteria Berdasarkan AHP')
            for i, w in enumerate(weights):
                st.write(f'{criteria_labels[i]}: {w:.4f}')

            st.subheader('Consistency Index (CI) dan Consistency Ratio (CR)')
            st.write(f'Consistency Index (CI): {CI:.4f}')
            st.write(f'Consistency Ratio (CR): {CR:.4f}')

            st.subheader('Perhitungan Skor AHP untuk Setiap Alternatif')
            st.dataframe(df_ahp[['Alternatif', 'Skor AHP']])

    else:
        st.warning('Silakan generate data terlebih dahulu di halaman Generate Data Dummy.')

elif selection == 'Perhitungan dan Komparasi Algoritma':
    st.title('Perhitungan dan Komparasi Algoritma')

    if 'ahp_weights' in st.session_state and 'df_ahp' in st.session_state:
        df_ahp = st.session_state['df_ahp']
        weights = st.session_state['ahp_weights']

        st.subheader('Data Alternatif dan Kriteria')
        st.dataframe(df_ahp)

        st.subheader('Bobot Kriteria')
        criteria_labels = df_ahp.columns[1:-1].tolist()
        for label, weight in zip(criteria_labels, weights):
            st.write(f'{label}: {weight:.4f}')

        st.subheader('Perhitungan Menggunakan AHP')
        st.dataframe(df_ahp[['Alternatif', 'Skor AHP']])

        # Menampilkan hasil WP
        st.subheader('Perhitungan Menggunakan WP')
        df_wp = wp_method(df_ahp.copy(), weights)
        if 'Skor WP' in df_wp.columns:
            st.dataframe(df_wp[['Alternatif', 'Skor WP']])
        else:
            st.error("Metode WP gagal menghitung skor.")

        # Menampilkan hasil SAW
        st.subheader('Perhitungan Menggunakan SAW')
        df_saw = saw_method(df_ahp.copy(), weights)
        if 'Skor SAW' in df_saw.columns:
            st.dataframe(df_saw[['Alternatif', 'Skor SAW']])
        else:
            st.error("Metode SAW gagal menghitung skor.")

        # Menampilkan hasil TOPSIS
        st.subheader('Perhitungan Menggunakan TOPSIS')
        df_topsis = topsis_method(df_ahp.copy(), weights)
        if 'Skor TOPSIS' in df_topsis.columns:
            st.dataframe(df_topsis[['Alternatif', 'Skor TOPSIS']])
        else:
            st.error("Metode TOPSIS gagal menghitung skor.")

    else:
        st.warning('Silakan lakukan perhitungan AHP terlebih dahulu untuk mendapatkan bobot kriteria.')

elif selection == 'Hasil Terbaik':
    st.title('Hasil Terbaik')

    if 'ahp_weights' in st.session_state and 'df_ahp' in st.session_state:
        df_ahp = st.session_state['df_ahp']
        weights = st.session_state['ahp_weights']

        st.subheader('Bobot Kriteria')
        criteria_labels = df_ahp.columns[1:-1].tolist()
        for label, weight in zip(criteria_labels, weights):
            st.write(f'{label}: {weight:.4f}')

        methods = ['AHP', 'WP', 'SAW', 'TOPSIS']
        best_alternatives = {}

        for method in methods:
            if method == 'AHP':
                df = df_ahp
                score_column = 'Skor AHP'
            elif method == 'WP':
                df = wp_method(df_ahp.copy(), weights)
                score_column = 'Skor WP'
            elif method == 'SAW':
                df = saw_method(df_ahp.copy(), weights)
                score_column = 'Skor SAW'
            elif method == 'TOPSIS':
                df = topsis_method(df_ahp.copy(), weights)
                score_column = 'Skor TOPSIS'

            if score_column in df.columns:
                best_alternatives[method] = df.loc[df[score_column].idxmax(), 'Alternatif']
            else:
                st.error(f"Metode {method} gagal menghitung skor.")

        st.subheader('Hasil Terbaik dari Setiap Metode')
        for method, best_alt in best_alternatives.items():
            st.write(f"Alternatif terbaik berdasarkan {method}: **{best_alt}**")

    else:
        st.warning('Silakan lakukan perhitungan AHP terlebih dahulu untuk mendapatkan bobot kriteria.')
