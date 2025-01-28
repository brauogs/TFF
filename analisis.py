import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, storage
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import io
import random

st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["firebase"])
    firebase_admin.initialize_app(cred, {'storageBucket': "vibraciones-aac24.appspot.com"})

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
        return [blob.name.split('/')[-1] for blob in blobs if blob.name != f"users/{user_id}/"]
    except Exception as e:
        st.error(f"Error al obtener los archivos del usuario: {str(e)}")
        return []

def download_file(user_id, file_name):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file_name}")
        return blob.download_as_bytes()
    except Exception as e:
        st.error(f"Error al descargar el archivo: {str(e)}")
        return None

# Signal processing functions
def corregir_linea_base(datos):
    return datos - np.mean(datos)

def aplicar_filtro_pasabanda(datos, fs, fmin=0.05, fmax=10):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [fmin/nyq, fmax/nyq], btype='band')
    return signal.filtfilt(b, a, datos)

def calcular_espectro_fourier(datos, fs):
    n = len(datos)
    frecuencias = fftfreq(n, d=1/fs)[:n//2]
    amplitudes = np.abs(fft(datos))[:n//2] * 2 / n
    return frecuencias, amplitudes

def dividir_entre_gravedad(x, y, z):
    return x / 9.81, y / 9.81, z / 9.81

def analisis_hv_mejorado(x, y, z, fs, num_ventanas=20, tamano_ventana=2000, suavizado=True):
    try:
        # Verificar longitud mínima de los datos
        if len(x) < tamano_ventana:
            raise ValueError("La señal es más corta que el tamaño de ventana especificado")

        # 1. Corrección de línea base y filtrado
        x = aplicar_filtro_pasabanda(corregir_linea_base(x), fs)
        y = aplicar_filtro_pasabanda(corregir_linea_base(y), fs)
        z = aplicar_filtro_pasabanda(corregir_linea_base(z), fs)

        # 2. Evitar división por cero en componentes Z
        z = np.where(z == 0, 1e-6, z)  # Reemplazar ceros por un valor pequeño

        # 3. Inicializar listas para almacenar HV
        hv_ratios = []

        # 4. Procesamiento por ventanas
        for _ in range(num_ventanas):
            # Seleccionar ventana aleatoria
            nini = random.randint(0, len(x) - tamano_ventana)
            x_win = x[nini:nini+tamano_ventana]
            y_win = y[nini:nini+tamano_ventana]
            z_win = z[nini:nini+tamano_ventana]

            # Calcular espectros
            frecuencias, fx = calcular_espectro_fourier(x_win, fs)
            _, fy = calcular_espectro_fourier(y_win, fs)
            _, fz = calcular_espectro_fourier(z_win, fs)

            # Calcular HV según método Nakamura (H = sqrt(HN^2 + HE^2))
            with np.errstate(divide='ignore', invalid='ignore'):
                hv = np.sqrt(fx**2 + fy**2) / fz
                hv = np.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)  # Manejar valores inválidos

            hv_ratios.append(hv)

        # 5. Promedio y desviación estándar
        hv_promedio = np.mean(hv_ratios, axis=0)
        hv_std = np.std(hv_ratios, axis=0)

        # 6. Suavizado mejorado
        if suavizado:
            hv_suavizado = signal.savgol_filter(hv_promedio, 
                                               window_length=min(11, len(hv_promedio)//2*2-1),  # Asegurar ventana impar
                                               polyorder=3)
        else:
            hv_suavizado = hv_promedio

        # 7. Detección robusta de pico fundamental
        # Restringir búsqueda a frecuencias relevantes (0.1 a 10 Hz)
        mask = (frecuencias >= 0.1) & (frecuencias <= 10)
        frecuencias_filtradas = frecuencias[mask]
        hv_filtrado = hv_suavizado[mask]

        # Encontrar pico más prominente
        peaks, _ = signal.find_peaks(hv_filtrado, prominence=0.5)
        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(hv_filtrado[peaks])]
            frecuencia_fundamental = frecuencias_filtradas[peak_idx]
        else:
            frecuencia_fundamental = frecuencias_filtradas[np.argmax(hv_filtrado)]

        periodo_fundamental = 1 / frecuencia_fundamental if frecuencia_fundamental > 0 else 0

        return {
            'frecuencias': frecuencias,
            'hv': hv_promedio,
            'hv_suavizado': hv_suavizado,
            'hv_mas_std': hv_promedio + hv_std,
            'hv_menos_std': hv_promedio - hv_std,
            'frecuencia_fundamental': frecuencia_fundamental,
            'periodo_fundamental': periodo_fundamental
        }

    except Exception as e:
        st.error(f"Error en el análisis H/V: {str(e)}")
        return None

# Visualization functions
def graficar_canales_individuales(x, y, z, fs, st, device_type='accelerometer'):
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Canal X', 'Canal Y', 'Canal Z'))
    
    tiempo = np.arange(len(x)) / fs
    line_width = 1 if device_type == 'accelerometer' else 3
    
    fig.add_trace(go.Scatter(x=tiempo, y=x, name='X', line=dict(width=line_width)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=y, name='Y', line=dict(width=line_width)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=z, name='Z', line=dict(width=line_width)), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Canales filtrados (0.05-10 Hz) - {device_type.title()}")
    
    img_bytes = fig.to_image(format="png")
    st.download_button(label="Descargar gráfica como imagen", data=img_bytes, file_name="grafica_canales.png", mime="image/png")
    return fig

def graficar_hv(resultados_hv, st):
    fig = go.Figure()
    
    # Curva H/V
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado'],
        mode='lines',
        name='H/V Suavizado',
        line=dict(color='blue', width=2)
    ))
    
    # Banda de desviación estándar
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_mas_std'],
        mode='lines',
        name='+1σ',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_menos_std'],
        mode='lines',
        name='-1σ',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    # Marcar frecuencia fundamental
    fig.add_trace(go.Scatter(
        x=[resultados_hv['frecuencia_fundamental']],
        y=[np.max(resultados_hv['hv_suavizado'])],
        mode='markers',
        name='Frecuencia Fundamental',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    # Configuración del gráfico
    fig.update_layout(
        title="Análisis H/V - Método Nakamura",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Amplitud H/V",
        xaxis_type="log",
        yaxis_type="linear",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig)
    return fig

# Main function
def main():
    st.title("Análisis del Acelerograma")
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
        st.write(f"Bienvenido, {st.session_state.user.email}")
        
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
            file_content = download_file(st.session_state.user.uid, selected_file)
            if file_content:
                st.download_button(label="Descargar archivo seleccionado", data=file_content, file_name=selected_file, mime="text/csv")
            
            df = pd.read_csv(io.BytesIO(file_content))
            st.write("Visualizar datos:")
            st.write(df.head())

            columnas_existentes = df.columns.tolist()
            st.write("Columnas en el archivo:", columnas_existentes)

            # Parámetros de análisis
            st.sidebar.header("Parámetros de análisis")
            fs = st.sidebar.number_input("Frecuencia de muestreo (Hz)", min_value=1, value=100)
            num_ventanas = st.sidebar.number_input("Número de ventanas para análisis H/V", min_value=1, max_value=100, value=20)
            tamano_ventana = st.sidebar.number_input("Tamaño de ventana (puntos)", min_value=100, max_value=10000, value=2000)
            dividir_por_g = st.sidebar.checkbox("Dividir datos por 9.81 (gravedad)", value=False)
            
            device_type = st.sidebar.selectbox(
                "Tipo de dispositivo",
                ["accelerometer", "mobile"],
                format_func=lambda x: "Acelerómetro" if x == "accelerometer" else "Dispositivo móvil"
            )

            suavizar_hv = st.sidebar.checkbox("Suavizar curva H/V", value=True)

            if st.button("Analizar datos"):
                datos_x = df['x'].values
                datos_y = df['y'].values
                datos_z = df['z'].values

                if dividir_por_g:
                    datos_x, datos_y, datos_z = dividir_entre_gravedad(datos_x, datos_y, datos_z)
                    
                    df_dividido = df.copy()
                    df_dividido['x'] = datos_x
                    df_dividido['y'] = datos_y
                    df_dividido['z'] = datos_z
                    
                    st.subheader("Comparación de datos originales y divididos entre 9.81")
                    df_comparacion = pd.concat([df.head(10), df_dividido.head(10)], axis=1)
                    df_comparacion.columns = ['x_original', 'y_original', 'z_original', 'x_dividido', 'y_dividido', 'z_dividido']
                    st.write(df_comparacion)
                
                resultados_hv = analisis_hv_mejorado(
                    datos_x, datos_y, datos_z,
                    fs=fs,
                    num_ventanas=num_ventanas,
                    tamano_ventana=tamano_ventana,
                    suavizado=suavizar_hv
                )
                
                st.subheader("Canales filtrados (0.05-10 Hz)")
                fig_canales = graficar_canales_individuales(
                    datos_x, datos_y, datos_z, fs, st, device_type
                )
                st.plotly_chart(fig_canales)
                
                st.subheader("Análisis H/V")
                fig_hv = graficar_hv(resultados_hv, st)
                st.plotly_chart(fig_hv)
                
                st.subheader("Estadísticas del análisis H/V")
                st.write(f"Frecuencia fundamental (X/Z): {resultados_hv['frecuencia_fundamental_xz']:.2f} Hz")
                st.write(f"Frecuencia fundamental (Y/Z): {resultados_hv['frecuencia_fundamental_yz']:.2f} Hz")

                if abs(resultados_hv['frecuencia_fundamental_xz'] - 1.16) <= 0.05:
                    st.success("La frecuencia fundamental (X/Z) coincide con el valor esperado de 1.16 Hz")
                else:
                    st.info(f"La frecuencia fundamental (X/Z) calculada ({resultados_hv['frecuencia_fundamental_xz']:.2f} Hz) difiere del valor esperado (1.16 Hz)")

                if abs(resultados_hv['frecuencia_fundamental_yz'] - 1.16) <= 0.05:
                    st.success("La frecuencia fundamental (Y/Z) coincide con el valor esperado de 1.16 Hz")
                else:
                    st.info(f"La frecuencia fundamental (Y/Z) calculada ({resultados_hv['frecuencia_fundamental_yz']:.2f} Hz) difiere del valor esperado (1.16 Hz)")

    st.sidebar.header("Instrucciones")
    st.sidebar.markdown("""
    1. Inicie sesión o cree una cuenta.
    2. Suba un archivo CSV o TXT para analizar.
    3. Seleccione un archivo de sus archivos subidos.
    4. Ajuste los parámetros de análisis en la barra lateral:
       - Frecuencia de muestreo
       - Número de ventanas para análisis H/V
       - Tamaño de ventana
       - Tipo de dispositivo
       - Suavizado de la curva H/V
    5. Haga clic en 'Analizar datos' para ver:
       - Canales filtrados individualmente
       - Análisis H/V
       - Estadísticas del análisis
    """)

if __name__ == "__main__":
    main()

