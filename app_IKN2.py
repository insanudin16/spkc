import streamlit as st
import pandas as pd
import numpy as np

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

# Normalization function for SAW
def saw_normalization(df, criteria_types):
    norm_df = df.copy()
    for col, criteria_type in zip(df.columns[1:], criteria_types):
        if criteria_type == 'benefit':
            norm_df[col] = df[col] / df[col].max()
        elif criteria_type == 'cost':
            norm_df[col] = df[col].min() / df[col]
    return norm_df

# TOPSIS calculation
def topsis(df, weights, criteria_types):
    norm_df = saw_normalization(df, criteria_types)
    weighted_df = norm_df.iloc[:, 1:].mul(weights, axis=1)
    
    ideal_best = []
    ideal_worst = []
    for col, criteria_type in zip(weighted_df.columns, criteria_types):
        if criteria_type == 'benefit':
            ideal_best.append(weighted_df[col].max())
            ideal_worst.append(weighted_df[col].min())
        elif criteria_type == 'cost':
            ideal_best.append(weighted_df[col].min())
            ideal_worst.append(weighted_df[col].max())

    df['D+'] = np.sqrt(((weighted_df - ideal_best) ** 2).sum(axis=1))
    df['D-'] = np.sqrt(((weighted_df - ideal_worst) ** 2).sum(axis=1))
    
    df['TOPSIS Score'] = df['D-'] / (df['D+'] + df['D-'])
    
    return df

# Streamlit App Layout
st.sidebar.title('Navigasi')
selection = st.sidebar.selectbox('Pilih Halaman', ['Home', 'Generate Data Dummy', 'Upload/Edit Data Alternatif', 'Input Bobot', 'Perhitungan WP', 'Perhitungan SAW', 'Perhitungan TOPSIS', 'Hasil Terbaik'])

if 'data_dummy' not in st.session_state:
    st.session_state['data_dummy'] = pd.DataFrame()

if selection == 'Home':
    st.title('Pemilihan Lahan di IKN Menggunakan Metode MCDM')
    st.write("""
    Aplikasi ini menggunakan metode MCDM seperti WP, SAW, dan TOPSIS untuk memilih lahan terbaik berdasarkan kriteria yang ditentukan.
    """)

elif selection == 'Generate Data Dummy':
    st.title('Generate Data Dummy Alternatif Lahan')
    
    num_data = st.number_input('Jumlah data yang ingin di-generate', min_value=1, max_value=1000, value=10)

    if st.button('Generate Data'):
        df_dummy = generate_dummy_data(num_data)
        st.session_state['data_dummy'] = df_dummy
        st.subheader('Data Alternatif Lahan yang Dihasilkan')
        st.dataframe(df_dummy)
        st.success(f'{num_data} baris data berhasil di-generate!')

elif selection == 'Upload/Edit Data Alternatif':
    st.title('Edit Data Alternatif Lahan')

    # Cek apakah data sudah di-generate sebelumnya
    if not st.session_state['data_dummy'].empty:
        st.subheader('Data Alternatif Lahan')
        st.dataframe(st.session_state['data_dummy'])
    else:
        st.warning('Silakan generate data dummy terlebih dahulu.')

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
            st.success('Bobot berhasil disimpan. Silakan lanjutkan ke perhitungan.')
    else:
        st.warning('Silakan unggah atau generate data terlebih dahulu.')

elif selection == 'Perhitungan WP':
    st.title('Perhitungan Metode WP')
    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']
        normalized_weights = [w / sum(weights) for w in weights]
        
        df_dummy['Skor WP'] = (
            (df_dummy['C1 (Luas Lahan)'] ** normalized_weights[0]) *
            (df_dummy['C2 (Ketersediaan Air)'] ** normalized_weights[1]) *
            (df_dummy['C3 (Kedekatan Infrastruktur)'] ** normalized_weights[2]) *
            (df_dummy['C4 (Biaya Perolehan)'] ** -normalized_weights[3])  # Cost is minimized
        )
        
        st.subheader('Hasil Perhitungan Skor WP')
        st.dataframe(df_dummy[['Alternatif', 'Skor WP']])
    else:
        st.warning('Silakan masukkan bobot dan data terlebih dahulu.')

elif selection == 'Perhitungan SAW':
    st.title('Perhitungan Metode SAW')
    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']
        criteria_types = ['benefit', 'benefit', 'benefit', 'cost']  # Define the type for each criterion
        
        norm_df = saw_normalization(df_dummy, criteria_types)
        norm_df['Skor SAW'] = norm_df.iloc[:, 1:].mul(weights).sum(axis=1)
        
        st.subheader('Hasil Perhitungan Skor SAW')
        st.dataframe(norm_df[['Alternatif', 'Skor SAW']])
    else:
        st.warning('Silakan masukkan bobot dan data terlebih dahulu.')

elif selection == 'Perhitungan TOPSIS':
    st.title('Perhitungan Metode TOPSIS')
    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']
        criteria_types = ['benefit', 'benefit', 'benefit', 'cost']
        
        topsis_df = topsis(df_dummy, weights, criteria_types)
        
        st.subheader('Hasil Perhitungan Skor TOPSIS')
        st.dataframe(topsis_df[['Alternatif', 'TOPSIS Score']])
    else:
        st.warning('Silakan masukkan bobot dan data terlebih dahulu.')

elif selection == 'Hasil Terbaik':
    st.title('Hasil Terbaik Berdasarkan Metode')
    if not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        best_wp = df_dummy.loc[df_dummy['Skor WP'].idxmax()]
        best_saw = df_dummy.loc[df_dummy['Skor SAW'].idxmax()]
        best_topsis = df_dummy.loc[df_dummy['TOPSIS Score'].idxmax()]
        
        st.write('### Lahan Terbaik Berdasarkan WP:')
        st.write(best_wp)
        st.write('### Lahan Terbaik Berdasarkan SAW:')
        st.write(best_saw)
        st.write('### Lahan Terbaik Berdasarkan TOPSIS:')
        st.write(best_topsis)
    else:
        st.warning('Silakan lakukan perhitungan terlebih dahulu.')
