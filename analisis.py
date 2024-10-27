import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import pyperclip
import random
import firebase_admin
from firebase_admin import credentials, auth, storage
import tempfile

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\braug\Desktop\TFF\vibraciones-aac24-firebase-adminsdk-6cx89-96f2ad5ea5.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': "vibraciones-aac24.appspot.com"
    })

# Authentication functions
def sign_up(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user
    except Exception as e:
        st.error(f"Error during sign up: {str(e)}")
        return None

def sign_in(email, password):
    try:
        user = auth.get_user_by_email(email)
        # In a real application, you would verify the password here
        return user
    except Exception as e:
        st.error(f"Error during sign in: {str(e)}")
        return None

# File storage functions
def upload_file(file, user_id):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file.name}")
        blob.upload_from_string(file.getvalue(), content_type=file.type)
        return blob.public_url
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def get_user_files(user_id):
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f"users/{user_id}/")
        return [blob.name.split('/')[-1] for blob in blobs]
    except Exception as e:
        st.error(f"Error retrieving user files: {str(e)}")
        return []

# Existing data analysis functions
def aplicar_filtro_pasabanda(datos, corte_bajo, corte_alto, fs, orden=5):
    nyq = 0.5 * fs
    bajo = corte_bajo / nyq
    alto = corte_alto / nyq
    b, a = signal.butter(orden, [bajo, alto], btype='band')
    return signal.filtfilt(b, a, datos)

def aplicar_taper(datos, porcentaje=5):
    taper = signal.windows.tukey(len(datos), alpha=porcentaje/100)
    return datos * taper

def calcular_fft(datos, fs):
    n = len(datos)
    resultado_fft = fft(datos)
    frecuencias = np.fft.fftfreq(n, 1/fs)[:n//2]
    magnitudes = 2.0/n * np.abs(resultado_fft[0:n//2])
    return frecuencias, magnitudes

# ... (include all other existing functions here)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")
    st.title("Análisis del Acelerograma")

    # Authentication
    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign In"):
                user = sign_in(email, password)
                if user:
                    st.session_state.user = user
                    st.success("Signed in successfully!")
                    st.experimental_rerun()

        with tab2:
            new_email = st.text_input("New Email")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                user = sign_up(new_email, new_password)
                if user:
                    st.session_state.user = user
                    st.success("Signed up successfully!")
                    st.experimental_rerun()
    else:
        st.write(f"Welcome, {st.session_state.user.email}")
        if st.button("Sign Out"):
            st.session_state.user = None
            st.experimental_rerun()

        # File upload and storage
        uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])
        if uploaded_file:
            file_url = upload_file(uploaded_file, st.session_state.user.uid)
            if file_url:
                st.success("File uploaded successfully!")

        # Display user's files
        user_files = get_user_files(st.session_state.user.uid)
        selected_file = st.selectbox("Select a file for analysis", user_files)

        if selected_file:
            # Download the selected file
            bucket = storage.bucket()
            blob = bucket.blob(f"users/{st.session_state.user.uid}/{selected_file}")
            _, temp_local_filename = tempfile.mkstemp()
            blob.download_to_filename(temp_local_filename)

            # Read the file
            df = pd.read_csv(temp_local_filename)
            os.remove(temp_local_filename)  # Clean up the temp file

            # Existing data analysis code
            st.write("Visualizar datos:")
            st.write(df.head())

            columnas_existentes = df.columns.tolist()
            st.write("Columnas en el archivo:", columnas_existentes)

            opciones_canal = ['x', 'y', 'z', 'Todos los canales']
            canal_seleccionado = st.selectbox("Selecciona el canal a analizar", opciones_canal)

            st.sidebar.header("Parámetros de filtrado")
            corte_bajo = st.sidebar.slider("Fmin (Hz)", 0.1, 10.0, 0.1, 0.1)
            corte_alto = st.sidebar.slider("Fmax (Hz)", 1.0, 50.0, 10.0, 0.1)
            porcentaje_taper = st.sidebar.slider("Taper (%)", 1, 20, 5, 1)

            num_rutinas_fft = st.number_input("Número de rutinas FFT a realizar", min_value=1, max_value=10, value=5)

            if st.button("Analizar datos"):
                canales = ['x', 'y', 'z'] if canal_seleccionado == 'Todos los canales' else [canal_seleccionado]
                resultados, fs, columna_tiempo = procesar_datos_sismicos(df, canales, corte_bajo, corte_alto, porcentaje_taper)
                fig = graficar_resultados(resultados, fs, canales)
                st.plotly_chart(fig)

                # ... (include all other existing analysis code here)

    st.sidebar.header("Instrucciones")
    st.sidebar.markdown("""
    1. Sign in or create an account.
    2. Upload CSV or TXT files for analysis.
    3. Select a file from your uploaded files.
    4. Choose the channel to analyze.
    5. Adjust filtering parameters.
    6. Specify the number of FFT routines.
    7. Click 'Analyze data' to view results.
    8. Download processed data if desired.
    """)

if __name__ == "__main__":
    main()