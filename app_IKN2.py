import streamlit as st
import pandas as pd
import numpy as np

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
    # Normalize the data
    norm_df = saw_normalization(df, criteria_types)
    
    # Weight the normalized data
    weighted_df = norm_df.iloc[:, 1:].mul(weights, axis=1)
    
    # Identify ideal best and ideal worst
    ideal_best = []
    ideal_worst = []
    for col, criteria_type in zip(weighted_df.columns, criteria_types):
        if criteria_type == 'benefit':
            ideal_best.append(weighted_df[col].max())
            ideal_worst.append(weighted_df[col].min())
        elif criteria_type == 'cost':
            ideal_best.append(weighted_df[col].min())
            ideal_worst.append(weighted_df[col].max())

    # Calculate the distance from ideal best and worst
    df['D+'] = np.sqrt(((weighted_df - ideal_best) ** 2).sum(axis=1))
    df['D-'] = np.sqrt(((weighted_df - ideal_worst) ** 2).sum(axis=1))
    
    # Calculate the TOPSIS score (closer to ideal best)
    df['TOPSIS Score'] = df['D-'] / (df['D+'] + df['D-'])
    
    return df

# Streamlit App Layout
st.sidebar.title('Navigasi')
selection = st.sidebar.selectbox('Pilih Halaman', ['Home', 'Generate Data Dummy', 'Input/Edit Data Alternatif', 'Input Bobot', 'Perhitungan WP', 'Perhitungan SAW', 'Perhitungan TOPSIS', 'Hasil Terbaik'])

if 'data_dummy' not in st.session_state:
    st.session_state['data_dummy'] = pd.DataFrame()

if selection == 'Home':
    st.title('Pemilihan Lahan di IKN Menggunakan Berbagai Metode MCDM')
    st.write("""
    Aplikasi ini menggunakan beberapa metode MCDM (Multiple Criteria Decision Making) seperti Weighted Product (WP), Simple Additive Weighting (SAW), dan TOPSIS untuk memilih lahan terbaik berdasarkan berbagai kriteria.
    Anda dapat memasukkan data alternatif lahan, bobot untuk setiap kriteria, dan melihat perhitungan serta hasilnya.
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
        st.success(f'{num_data} baris data berhasil di-generate dan siap untuk di-download!')

elif selection == 'Input/Edit Data Alternatif':
    st.title('Input/Edit Data Alternatif Lahan')
    if not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        st.subheader('Edit Data Alternatif Lahan')
        try:
            edited_df = st.experimental_data_editor(df_dummy, num_rows="dynamic")
        except Exception as e:
            st.error(f"Error terjadi saat mengedit data: {str(e)}")
            st.stop()
        if st.button('Simpan Perubahan'):
            st.session_state['data_dummy'] = edited_df
            st.success('Data Alternatif berhasil diperbarui!')
    else:
        st.warning('Silakan generate atau upload data terlebih dahulu di halaman Generate Data Dummy.')

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
            st.success('Bobot berhasil disimpan. Silakan lanjutkan ke bagian perhitungan.')
    else:
        st.warning('Silakan generate atau upload data terlebih dahulu di halaman Generate Data Dummy.')

elif selection == 'Perhitungan WP':
    st.title('Perhitungan Metode Weighted Product (WP)')
    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']
        st.write(f'Bobot yang digunakan: {weights}')
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
        criteria_types = ['benefit', 'benefit', 'benefit', 'cost']  # Define the type for each criterion
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