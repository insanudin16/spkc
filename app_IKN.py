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

# Streamlit App Layout
st.sidebar.title('Navigasi')
selection = st.sidebar.selectbox('Pilih Halaman', ['Home', 'Generate Data Dummy', 'Input/Edit Data Alternatif', 'Input Bobot', 'Perhitungan WP', 'Hasil Terbaik'])

if 'data_dummy' not in st.session_state:
    st.session_state['data_dummy'] = pd.DataFrame()

if selection == 'Home':
    st.title('Pemilihan Lahan di IKN Menggunakan Metode WP')
    st.write("""
    Aplikasi ini menggunakan metode Weighted Product (WP) untuk memilih lahan terbaik berdasarkan berbagai kriteria. 
    Anda dapat memasukkan data alternatif lahan, bobot untuk setiap kriteria, dan melihat perhitungan serta hasilnya.
    """)

elif selection == 'Generate Data Dummy':
    st.title('Generate Dummy Data Alternatif Lahan')
    
    # Input to determine the number of data points
    num_data = st.number_input('Jumlah data yang ingin di-generate', min_value=1, max_value=1000, value=10)

    # Button to generate data
    if st.button('Generate Data'):
        # Generate the dummy data
        df_dummy = generate_dummy_data(num_data)
        
        # Save to session state
        st.session_state['data_dummy'] = df_dummy
        
        # Display the generated data
        st.subheader('Data Alternatif Lahan yang Dihasilkan')
        st.dataframe(df_dummy)
        
        # Save the generated data to CSV
        csv = df_dummy.to_csv(index=False)
        st.download_button(label="Download Data sebagai CSV", data=csv, file_name='data_alternatif_lahan.csv', mime='text/csv')

        st.success(f'{num_data} baris data berhasil di-generate dan siap untuk di-download!')

elif selection == 'Input/Edit Data Alternatif':
    st.title('Input/Edit Data Alternatif Lahan')

    # Check if data is available in session state
    if not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']

        # Display the current dataset with editable fields
        st.subheader('Edit Data Alternatif Lahan')
        edited_df = st.experimental_data_editor(df_dummy, num_rows="dynamic")

        # Save changes back to session state
        if st.button('Simpan Perubahan'):
            st.session_state['data_dummy'] = edited_df
            st.success('Data Alternatif berhasil diperbarui!')
    else:
        st.warning('Silakan generate atau upload data terlebih dahulu di halaman Generate Data Dummy.')

elif selection == 'Input Bobot':
    st.title('Input Bobot Kriteria')

    # Check if data is available
    if not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        
        # Display the dataset
        st.subheader('Data Alternatif Lahan')
        st.dataframe(df_dummy)

        # Criteria Weights Input
        st.subheader('Input Bobot Kriteria')
        w1 = st.number_input('Bobot C1 (Luas Lahan)', min_value=0, max_value=10, value=5)
        w2 = st.number_input('Bobot C2 (Ketersediaan Air)', min_value=0, max_value=10, value=3)
        w3 = st.number_input('Bobot C3 (Kedekatan Infrastruktur)', min_value=0, max_value=10, value=4)
        w4 = st.number_input('Bobot C4 (Biaya Perolehan)', min_value=0, max_value=10, value=2)

        # Save the weights in session state to be used in next steps
        if st.button('Simpan Bobot'):
            st.session_state['weights'] = [w1, w2, w3, w4]
            st.success('Bobot berhasil disimpan. Silakan lanjutkan ke bagian Perhitungan WP.')
    else:
        st.warning('Silakan generate atau upload data terlebih dahulu di halaman Generate Data Dummy.')

elif selection == 'Perhitungan WP':
    st.title('Perhitungan Metode Weighted Product (WP)')
    
    # Check if data and weights are available
    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']
        st.write(f'Bobot yang digunakan: {weights}')
        
        # Normalize the weights
        normalized_weights = [w / sum(weights) for w in weights]
        
        # Calculating the WP score
        df_dummy['Skor WP'] = (
            (df_dummy['C1 (Luas Lahan)'] ** normalized_weights[0]) *
            (df_dummy['C2 (Ketersediaan Air)'] ** normalized_weights[1]) *
            (df_dummy['C3 (Kedekatan Infrastruktur)'] ** normalized_weights[2]) *
            (df_dummy['C4 (Biaya Perolehan)'] ** -normalized_weights[3])  # For cost, negative exponent
        )
        
        # Display the results
        st.subheader('Hasil Perhitungan Skor WP')
        st.dataframe(df_dummy[['Alternatif', 'Skor WP']])
        
    else:
        st.warning('Silakan masukkan bobot dan data terlebih dahulu.')

elif selection == 'Hasil Terbaik':
    st.title('Alternatif Terbaik Berdasarkan Metode WP')

    # Check if data and weights are available
    if 'weights' in st.session_state and not st.session_state['data_dummy'].empty:
        df_dummy = st.session_state['data_dummy']
        weights = st.session_state['weights']
        normalized_weights = [w / sum(weights) for w in weights]
        
        # Calculate the WP score if not calculated
        if 'Skor WP' not in df_dummy.columns:
            df_dummy['Skor WP'] = (
                (df_dummy['C1 (Luas Lahan)'] ** normalized_weights[0]) *
                (df_dummy['C2 (Ketersediaan Air)'] ** normalized_weights[1]) *
                (df_dummy['C3 (Kedekatan Infrastruktur)'] ** normalized_weights[2]) *
                (df_dummy['C4 (Biaya Perolehan)'] ** -normalized_weights[3])
            )
        
        # Show the best alternative
        best_alternative = df_dummy.loc[df_dummy['Skor WP'].idxmax()]['Alternatif']
        st.success(f'Alternatif terbaik berdasarkan metode WP adalah: {best_alternative}')
    else:
        st.warning('Silakan lakukan perhitungan WP terlebih dahulu.')
