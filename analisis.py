import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, storage
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import io
import random

st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

# Firebase setup functions (unchanged)
def get_firebase_credentials():
    try:
        return dict(st.secrets["firebase"])
    except KeyError:
        st.error("Error: Las credenciales de Firebase no se encuentran en los secretos de Streamlit.")
        return None

if not firebase_admin._apps:
    firebase_cred = get_firebase_credentials()
    if firebase_cred:
        try:
            cred = credentials.Certificate(firebase_cred)
            firebase_admin.initialize_app(cred, {
                'storageBucket': "vibraciones-aac24.appspot.com"
            })
        except ValueError as e:
            st.error(f"Error al inicializar Firebase: {str(e)}")
    else:
        st.error("No se pudieron obtener las credenciales de Firebase.")

def sign_up(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user
    except Exception as e:
        st.error(f"Error durante el registro: {str(e)}")
        return None

def sign_in(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user
    except Exception as e:
        st.error(f"Error durante el inicio de sesión: {str(e)}")
        return None

def upload_file(file, user_id):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file.name}")
        blob.upload_from_string(file.getvalue(), content_type=file.type)
        return blob.public_url
    except Exception as e:
        st.error(f"Error al subir el archivo: {str(e)}")
        return None

def get_user_files(user_id):
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f"users/{user_id}/")
        user_files = [blob.name.split('/')[-1] for blob in blobs if blob.name != f"users/{user_id}/"]
        return user_files
    except Exception as e:
        st.error(f"Error al obtener los archivos del usuario: {str(e)}")
        return []

# Updated signal processing functions
def corregir_linea_base(datos):
    return datos - np.mean(datos)

def aplicar_filtro_pasabanda(datos, fs, corte_bajo=0.05, corte_alto=10, orden=4):
    nyq = 0.5 * fs
    bajo = corte_bajo / nyq
    alto = corte_alto / nyq
    b, a = signal.butter(orden, [bajo, alto], btype='band')
    return signal.filtfilt(b, a, datos)

def calcular_espectro_fourier(datos):
    return np.abs(fft(datos))

def analisis_hv(x, y, z, fs, num_ventanas=20, tamano_ventana=2000):
    cocientes_xz = []
    cocientes_yz = []
    
    for _ in range(num_ventanas):
        nini = random.randint(0, len(x) - tamano_ventana)
        x1 = x[nini:nini+tamano_ventana]
        y1 = y[nini:nini+tamano_ventana]
        z1 = z[nini:nini+tamano_ventana]
        
        fft_x1 = calcular_espectro_fourier(x1)
        fft_y1 = calcular_espectro_fourier(y1)
        fft_z1 = calcular_espectro_fourier(z1)
        
        cociente_xz = np.mean(fft_x1 / fft_z1)
        cociente_yz = np.mean(fft_y1 / fft_z1)
        
        cocientes_xz.append(cociente_xz)
        cocientes_yz.append(cociente_yz)
    
    return cocientes_xz, cocientes_yz

def procesar_datos_sismicos(df, canales, fs):
    resultados = {}
    for canal in canales:
        datos = df[canal].values
        datos_corregidos = corregir_linea_base(datos)
        datos_filtrados = aplicar_filtro_pasabanda(datos_corregidos, fs)
        resultados[canal] = datos_filtrados
    return resultados

def graficar_resultados(resultados, fs, canales):
    fig = make_subplots(rows=len(canales), cols=2, 
                        subplot_titles=[f"Canal {canal.upper()} (Filtrado)" for canal in canales] + 
                                       [f"Canal {canal.upper()} FFT" for canal in canales],
                        vertical_spacing=0.1)

    for i, canal in enumerate(canales, start=1):
        tiempo = np.arange(len(resultados[canal])) / fs
        fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal], name=f"{canal.upper()} Filtrado"), row=i, col=1)
        
        frecuencias = np.fft.fftfreq(len(resultados[canal]), d=1/fs)[:len(resultados[canal])//2]
        amplitudes = np.abs(fft(resultados[canal]))[:len(resultados[canal])//2] * 2 / len(resultados[canal])
        fig.add_trace(go.Scatter(x=frecuencias, y=amplitudes, name=f"{canal.upper()} FFT"), row=i, col=2)

    fig.update_layout(height=300*len(canales), width=1200, title_text="Análisis de Canales")
    for i in range(1, len(canales)+1):
        fig.update_xaxes(title_text="Tiempo (s)", row=i, col=1)
        fig.update_xaxes(title_text="Frecuencia (Hz)", row=i, col=2)
        fig.update_yaxes(title_text="Amplitud", row=i, col=1)
        fig.update_yaxes(title_text="Magnitud", row=i, col=2)
    return fig

def guardar_estadisticas(cocientes_xz, cocientes_yz):
    datos = {
        'Cociente x/z': cocientes_xz,
        'Cociente y/z': cocientes_yz
    }
    df = pd.DataFrame(datos)
    return df.to_csv(index=False).encode('utf-8')

def descargar_datos_procesados(resultados, canales, fs):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for canal in canales:
            tiempo = np.arange(len(resultados[canal])) / fs
            df = pd.DataFrame({
                'Tiempo': tiempo,
                'Filtrado': resultados[canal]
            })
            df.to_excel(writer, sheet_name=f'Canal_{canal}', index=False)
    
    output.seek(0)
    return output

def main():
    st.title("Análisis del Acelerograma")

    # Sidebar image placeholder
    st.sidebar.image("logoUAMSis.png", use_container_width=True)

    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Iniciar sesión", "Registrarse"])
        
        with tab1:
            email = st.text_input("Correo electrónico")
            password = st.text_input("Contraseña", type="password")
            if st.button("Iniciar sesión"):
                user = sign_in(email, password)
                if user:
                    st.session_state.user = user
                    st.success("¡Sesión iniciada con éxito!")
                    st.rerun()

        with tab2:
            new_email = st.text_input("Nuevo correo electrónico")
            new_password = st.text_input("Nueva contraseña", type="password")
            if st.button("Registrarse"):
                user = sign_up(new_email, new_password)
                if user:
                    st.session_state.user = user
                    st.success("¡Registrado con éxito!")
                    st.rerun()
    else:
        user_email = st.session_state.user.email
        st.write(f"Bienvenido, {user_email}")
        
        if st.button("Cerrar sesión"):
            st.session_state.user = None
            st.rerun()

        uploaded_file = st.file_uploader("Sube un archivo CSV o TXT", type=["csv", "txt"])
        if uploaded_file:
            file_url = upload_file(uploaded_file, st.session_state.user.uid)
            if file_url:
                st.success("¡Archivo subido exitosamente!")

        user_files = get_user_files(st.session_state.user.uid)
        selected_file = st.selectbox("Seleccione un archivo para analizar", user_files)

        if selected_file:
            bucket = storage.bucket()
            blob = bucket.blob(f"users/{st.session_state.user.uid}/{selected_file}")
            _, temp_local_filename = tempfile.mkstemp()
            blob.download_to_filename(temp_local_filename)

            df = pd.read_csv(temp_local_filename)
            os.remove(temp_local_filename)

            st.write("Visualizar datos:")
            st.write(df.head())

            columnas_existentes = df.columns.tolist()
            st.write("Columnas en el archivo:", columnas_existentes)

            st.sidebar.header("Parámetros de análisis")
            fs = st.sidebar.number_input("Frecuencia de muestreo (Hz)", min_value=1, value=100)
            num_ventanas = st.sidebar.number_input("Número de ventanas para análisis H/V", min_value=1, max_value=100, value=20)
            tamano_ventana = st.sidebar.number_input("Tamaño de ventana (puntos)", min_value=100, max_value=10000, value=2000)

            if st.button("Analizar datos"):
                canales = ['x', 'y', 'z']
                resultados = procesar_datos_sismicos(df, canales, fs)
                fig = graficar_resultados(resultados, fs, canales)
                st.plotly_chart(fig)

                st.subheader("Análisis H/V")
                cocientes_xz, cocientes_yz = analisis_hv(resultados['x'], resultados['y'], resultados['z'], fs, num_ventanas, tamano_ventana)
                
                promedio_xz = np.mean(cocientes_xz)
                desviacion_xz = np.std(cocientes_xz)
                promedio_yz = np.mean(cocientes_yz)
                desviacion_yz = np.std(cocientes_yz)

                st.write("Resultados del análisis H/V:")
                st.write(f"Cociente x/z: Promedio = {promedio_xz:.4f}, Promedio + Desviación = {promedio_xz + desviacion_xz:.4f}, Promedio - Desviación = {promedio_xz - desviacion_xz:.4f}")
                st.write(f"Cociente y/z: Promedio = {promedio_yz:.4f}, Promedio + Desviación = {promedio_yz + desviacion_yz:.4f}, Promedio - Desviación = {promedio_yz - desviacion_yz:.4f}")

                fig_hv = go.Figure()
                fig_hv.add_trace(go.Scatter(y=cocientes_xz, name="Cociente x/z", mode='markers'))
                fig_hv.add_trace(go.Scatter(y=cocientes_yz, name="Cociente y/z", mode='markers'))
                fig_hv.update_layout(title="Cocientes H/V por ventana", xaxis_title="Número de ventana", yaxis_title="Cociente H/V")
                st.plotly_chart(fig_hv)

                # Generar y descargar estadísticas
                csv = guardar_estadisticas(cocientes_xz, cocientes_yz)
                st.download_button(
                    label="Descargar estadísticas H/V",
                    data=csv,
                    file_name="estadisticas_hv.csv",
                    mime="text/csv",
                )

                # Add download button for processed data
                output = descargar_datos_procesados(resultados, canales, fs)
                st.download_button(
                    label="Descargar datos procesados",
                    data=output,
                    file_name="datos_procesados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    st.sidebar.header("Instrucciones")
    st.sidebar.markdown("""
    1. Inicie sesión o cree una cuenta.
    2. Suba un archivo CSV o TXT para analizar.
    3. Seleccione un archivo de sus archivos subidos.
    4. Ajuste los parámetros de análisis en la barra lateral.
    5. Haga clic en 'Analizar datos' para ver los resultados.
    6. Revise los gráficos de los canales filtrados, FFT y análisis H/V.
    7. Descargue las estadísticas H/V si lo desea.
    """)

if __name__ == "__main__":
    main()

