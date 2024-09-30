import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Random Index (RI) values for different matrix sizes (used in AHP)
RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
           11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}

# Function to generate dummy data
def generate_dummy_data(num_data, num_criteria):
    data = {
        'Alternatif': [f'Lahan {i+1}' for i in range(num_data)]
    }
    for i in range(num_criteria):
        data[f'C{i+1}'] = np.random.randint(1, 100, size=num_data)
    df = pd.DataFrame(data)
    return df

# New function to generate consistent AHP matrices
def generate_consistent_ahp_matrix(n):
    """
    Generate a consistent AHP matrix for n criteria.
    """
    # Generate random weights
    weights = np.random.rand(n)
    weights /= weights.sum()
    
    # Create the consistent matrix
    matrix = np.outer(weights, 1/weights)
    
    # Round the matrix values to one decimal place
    matrix = np.round(matrix, decimals=1)
    
    return matrix

# Dictionary to store consistent matrices for 4-15 criteria
consistent_matrices = {i: generate_consistent_ahp_matrix(i) for i in range(4, 16)}

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
    RI = RI_dict[n]
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

# Streamlit App Layout
st.sidebar.title('Navigasi')
selection = st.sidebar.selectbox('Pilih Halaman', ['Home', 'Generate Data Dummy', 'Input/Edit Data Alternatif', 'Input Matriks Perbandingan Berpasangan (AHP)', 'Hasil AHP'])

if 'data_dummy' not in st.session_state:
    st.session_state['data_dummy'] = pd.DataFrame()
if 'num_criteria' not in st.session_state:
    st.session_state['num_criteria'] = 4

if selection == 'Home':
    st.title('Pemilihan Lahan di IKN Menggunakan Metode AHP')
    st.write("""
    Aplikasi ini menggunakan metode Analytic Hierarchy Process (AHP) untuk memilih lahan terbaik berdasarkan kriteria yang dapat disesuaikan. 
    Anda dapat meng-generate data dummy, mengedit data, memasukkan bobot, dan melakukan perhitungan dengan metode AHP.
    """)

elif selection == 'Generate Data Dummy':
    st.title('Generate Data Dummy Alternatif Lahan')
    
    num_data = st.number_input('Jumlah data yang ingin di-generate', min_value=1, max_value=1000, value=10)
    
    st.subheader('Kriteria')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Tambah Kriteria', key='add_criteria'):
            if st.session_state['num_criteria'] < 15:
                st.session_state['num_criteria'] += 1
    with col2:
        if st.button('Kurangi Kriteria', key='remove_criteria'):
            if st.session_state['num_criteria'] > 2:
                st.session_state['num_criteria'] -= 1
    
    st.write(f"Jumlah kriteria saat ini: {st.session_state['num_criteria']}")

    if st.button('Generate Data'):
        df_dummy = generate_dummy_data(num_data, st.session_state['num_criteria'])
        st.session_state['data_dummy'] = df_dummy
        st.subheader('Data Alternatif Lahan yang Dihasilkan')
        st.dataframe(df_dummy)
        csv = df_dummy.to_csv(index=False)
        st.download_button(label="Download Data sebagai CSV", data=csv, file_name='data_alternatif_lahan.csv', mime='text/csv')
        st.success(f'{num_data} baris data dengan {st.session_state["num_criteria"]} kriteria berhasil di-generate!')
        st.info('Data telah otomatis dimasukkan ke menu Input/Edit Data Alternatif untuk pengeditan lebih lanjut.')

elif selection == 'Input/Edit Data Alternatif':
    st.title('Input/Edit Data Alternatif Lahan')

    if st.session_state['data_dummy'].empty:
        st.warning('Belum ada data. Silakan generate data terlebih dahulu di halaman Generate Data Dummy.')
    else:
        df = st.session_state['data_dummy'].copy()
        
        st.subheader('Edit Data Alternatif Lahan')
        edited_df = st.data_editor(df)
        
        if st.button('Simpan Perubahan'):
            st.session_state['data_dummy'] = edited_df
            st.success('Data berhasil diperbarui!')

elif selection == 'Input Matriks Perbandingan Berpasangan (AHP)':
    st.title('Input Matriks Perbandingan Berpasangan untuk AHP')

    if not st.session_state['data_dummy'].empty:
        st.subheader('Data Alternatif Lahan')
        st.dataframe(st.session_state['data_dummy'])

        st.subheader('Matriks Perbandingan Berpasangan')
        num_criteria = st.session_state['num_criteria']
        criteria_labels = [f'C{i+1}' for i in range(num_criteria)]

        # Use default consistent matrix or custom input
        use_default = st.checkbox('Gunakan matriks konsisten default', value=True)

        if use_default:
            pairwise_matrix = consistent_matrices[num_criteria]
        else:
            pairwise_matrix = np.ones((num_criteria, num_criteria))
            for i in range(num_criteria):
                for j in range(i+1, num_criteria):
                    value = st.number_input(
                        f'Perbandingan {criteria_labels[i]} vs {criteria_labels[j]}',
                        min_value=0.1,
                        max_value=10.0,
                        value=float(pairwise_matrix[i, j]),
                        step=0.1,
                        format="%.1f",
                        key=f'comp_{i}_{j}'
                    )
                    pairwise_matrix[i, j] = value
                    pairwise_matrix[j, i] = 1 / value

        st.subheader('Matriks Perbandingan yang Digunakan')
        st.dataframe(pd.DataFrame(pairwise_matrix, index=criteria_labels, columns=criteria_labels))

        if st.button('Hitung AHP'):
            df_dummy = st.session_state['data_dummy']
            df_ahp, weights, CI, CR = ahp_method(df_dummy.copy(), pairwise_matrix)
            st.session_state['df_ahp'] = df_ahp
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

elif selection == 'Hasil AHP':
    st.title('Hasil AHP')

    if 'ahp_weights' in st.session_state and 'df_ahp' in st.session_state:
        df_ahp = st.session_state['df_ahp']
        weights = st.session_state['ahp_weights']

        st.subheader('Bobot Kriteria')
        criteria_labels = df_ahp.columns[1:-1].tolist()
        for label, weight in zip(criteria_labels, weights):
            st.write(f'{label}: {weight:.4f}')

        st.subheader('5 Alternatif Terbaik Berdasarkan AHP')
        top_5 = df_ahp.nlargest(5, 'Skor AHP')[['Alternatif', 'Skor AHP']]
        st.dataframe(top_5)

    else:
        st.warning('Silakan lakukan perhitungan AHP terlebih dahulu untuk mendapatkan hasil.')
