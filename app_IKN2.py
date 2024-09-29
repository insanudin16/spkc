import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Function to generate dummy data
def generate_dummy_data(num_data):
    # Generate random data for each criterion
    data = {
        'Alternatif': [f'Lahan {i+1}' for i in range(num_data)],
        'C1 (Luas Lahan)': np.random.randint(40, 100, size=num_data),  # Luas Lahan (e.g. 40 - 100 hektar)
        'C2 (Ketersediaan Air)': np.random.randint(10000, 20000, size=num_data),  # Ketersediaan Air (e.g. 10,000 - 20,000 liter)
        'C3 (Kedekatan Infrastruktur)': np.random.randint(5, 20, size=num_data),  # Kedekatan Infrastruktur (e.g. 5 - 20 km)
        'C4 (Biaya Perolehan)': np.random.randint(80, 150, size=num_data)  # Biaya Perolehan (e.g. 80 - 150 juta)
    }
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Function to normalize weights
def normalize_weights(weights):
    return [w / sum(weights) for w in weights]

# WP Calculation
def wp_method(df, weights):
    normalized_weights = normalize_weights(weights)
    df['Skor WP'] = (
        (df['C1 (Luas Lahan)'] ** normalized_weights[0]) *
        (df['C2 (Ketersediaan Air)'] ** normalized_weights[1]) *
        (df['C3 (Kedekatan Infrastruktur)'] ** normalized_weights[2]) *
        (df['C4 (Biaya Perolehan)'] ** -normalized_weights[3])  # For cost, negative exponent
    )
    return df

# SAW Calculation
def saw_method(df, weights):
    normalized_weights = normalize_weights(weights)
    # Normalize the criteria by dividing each value by the max value in its column
    for col in df.columns[1:]:
        if col != 'C4 (Biaya Perolehan)':  # Benefit criteria
            df[col] = df[col] / df[col].max()
        else:  # Cost criteria
            df[col] = df[col].min() / df[col]
    
    # Calculate the SAW score
    df['Skor SAW'] = df.iloc[:, 1:].dot(normalized_weights)
    return df

# AHP Calculation (simplified, without consistency ratio for simplicity)
def ahp_method(df, weights):
    normalized_weights = normalize_weights(weights)
    df['Skor AHP'] = df.iloc[:, 1:].dot(normalized_weights)
    return df

# TOPSIS Calculation
def topsis_method(df, weights):
    normalized_weights = normalize_weights(weights)
    
    # Normalize the decision matrix
    for i, col in enumerate(df.columns[1:]):
        df[col] = df[col] / np.sqrt((df[col] ** 2).sum())
    
    # Weighted normalized decision matrix
    for i, col in enumerate(df.columns[1:]):
        df[col] *= normalized_weights[i]
    
    # Determine positive and negative ideal solutions
    positive_ideal = []
    negative_ideal = []
    
    # Handle benefit and cost criteria
    for i, col in enumerate(df.columns[1:]):
        if 'Biaya' in col:  # If it's a cost criterion, minimum is ideal
            positive_ideal.append(df[col].min())
            negative_ideal.append(df[col].max())
        else:  # For benefit criteria, maximum is ideal
            positive_ideal.append(df[col].max())
            negative_ideal.append(df[col].min())
    
    # Convert to Series for proper subtraction
    positive_ideal = pd.Series(positive_ideal, index=df.columns[1:])
    negative_ideal = pd.Series(negative_ideal, index=df.columns[1:])
    
    # Calculate distances to positive and negative ideal solutions
    df['D+'] = np.sqrt(((df.iloc[:, 1:] - positive_ideal) ** 2).sum(axis=1))
    df['D-'] = np.sqrt(((df.iloc[:, 1:] - negative_ideal) ** 2).sum(axis=1))
    
    # Calculate TOPSIS score
    df['Skor TOPSIS'] = df['D-'] / (df['D-'] + df['D+'])
    return df

# Streamlit App Layout
st.sidebar.title('Navigasi')
selection = st.sidebar.selectbox('Pilih Halaman', ['Home', 'Generate Data Dummy', 'Input/Edit Data Alternatif', 'Input Bobot', 'Perhitungan dan Komparasi Algoritma', 'Hasil Terbaik'])

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

elif selection == 'Input/Edit Data Alternatif':
    st.title('Input/Edit Data Alternatif Lahan')

    if not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        st.subheader('Edit Data Alternatif Lahan')
        edited_df = st.experimental_data_editor(df_dummy, num_rows="dynamic")
        if st.button('Simpan Perubahan'):
            st.session_state['data_dummy'] = edited_df
            st.success('Data Alternatif berhasil diperbarui!')
    else:
        st.warning('Silakan generate data terlebih dahulu di halaman Generate Data Dummy.')

elif selection == 'Input Bobot':
    st.title('Input Bobot Kriteria')

    if not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        st.subheader('Data Alternatif Lahan')
        st.dataframe(df_dummy)

        st.subheader('Input Bobot Kriteria')
        w1 = st.number_input('Bobot C1 (Luas Lahan)', min_value=0, max_value=10, value=5)
        w2 = st.number_input('Bobot C2 (Ketersediaan Air)', min_value=0, max_value=10, value=3)
        w3 = st.number_input('Bobot C3 (Kedekatan Infrastruktur)', min_value=0, max_value=10, value=4)
        w4 = st.number_input('Bobot C4 (Biaya Perolehan)', min_value=0, max_value=10, value=2)

        if st.button('Simpan Bobot'):
            st.session_state['weights'] = [w1, w2, w3, w4]
            st.success('Bobot berhasil disimpan.')
    else:
        st.warning('Silakan generate data terlebih dahulu di halaman Generate Data Dummy.')

elif selection == 'Perhitungan dan Komparasi Algoritma':
    st.title('Perhitungan dan Komparasi Algoritma WP, SAW, AHP, dan TOPSIS')

    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']

        st.write(f'Alur perhitungan menggunakan bobot: {weights}')
        
        # WP
        st.subheader('Perhitungan WP')
        df_wp = wp_method(df_dummy.copy(), weights)
        st.dataframe(df_wp[['Alternatif', 'Skor WP']])

        # SAW
        st.subheader('Perhitungan SAW')
        df_saw = saw_method(df_dummy.copy(), weights)
        st.dataframe(df_saw[['Alternatif', 'Skor SAW']])

        # AHP
        st.subheader('Perhitungan AHP')
        df_ahp = ahp_method(df_dummy.copy(), weights)
        st.dataframe(df_ahp[['Alternatif', 'Skor AHP']])

        # TOPSIS
        st.subheader('Perhitungan TOPSIS')
        df_topsis = topsis_method(df_dummy.copy(), weights)
        st.dataframe(df_topsis[['Alternatif', 'Skor TOPSIS']])

        # Comparison table
        st.subheader('Perbandingan Skor dari Berbagai Algoritma')
        df_comparison = pd.DataFrame({
            'Alternatif': df_wp['Alternatif'],
            'WP': df_wp['Skor WP'],
            'SAW': df_saw['Skor SAW'],
            'AHP': df_ahp['Skor AHP'],
            'TOPSIS': df_topsis['Skor TOPSIS']
        })
        st.dataframe(df_comparison)
    else:
        st.warning('Silakan masukkan bobot terlebih dahulu di halaman Input Bobot.')

elif selection == 'Hasil Terbaik':
    st.title('Alternatif Terbaik Berdasarkan Algoritma WP, SAW, AHP, dan TOPSIS')

    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']

        # WP
        df_wp = wp_method(df_dummy.copy(), weights)
        best_wp = df_wp.loc[df_wp['Skor WP'].idxmax()]['Alternatif']

        # SAW
        df_saw = saw_method(df_dummy.copy(), weights)
        best_saw = df_saw.loc[df_saw['Skor SAW'].idxmax()]['Alternatif']

        # AHP
        df_ahp = ahp_method(df_dummy.copy(), weights)
        best_ahp = df_ahp.loc[df_ahp['Skor AHP'].idxmax()]['Alternatif']

        # TOPSIS
        df_topsis = topsis_method(df_dummy.copy(), weights)
        best_topsis = df_topsis.loc[df_topsis['Skor TOPSIS'].idxmax()]['Alternatif']

        st.subheader('Alternatif Terbaik Berdasarkan Setiap Algoritma')
        st.write(f"Algoritma WP: {best_wp}")
        st.write(f"Algoritma SAW: {best_saw}")
        st.write(f"Algoritma AHP: {best_ahp}")
        st.write(f"Algoritma TOPSIS: {best_topsis}")
    else:
        st.warning('Silakan lakukan perhitungan terlebih dahulu.')
