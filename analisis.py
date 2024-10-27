import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import firebase_admin 
from firebase_admin import credentials, auth, storage
import tempfile
import json
import os

st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

# Obtener credenciales de Firebase desde los secrets de Streamlit Cloud
def get_firebase_credentials():
    try:
        return dict(st.secrets["firebase"])  # Lee las credenciales desde los secrets en Streamlit Cloud
    except KeyError:
        st.error("Error: Las credenciales de Firebase no se encuentran en los secretos de Streamlit.")
        return None

# Inicializar Firebase
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

# Funciones de autenticación
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
        # Aquí se debería verificar la contraseña (esto es solo un ejemplo)
        return user
    except Exception as e:
        st.error(f"Error durante el inicio de sesión: {str(e)}")
        return None

# Función para subir archivos al almacenamiento
def upload_file(file, user_id):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file.name}")
        blob.upload_from_string(file.getvalue(), content_type=file.type)
        return blob.public_url
    except Exception as e:
        st.error(f"Error al subir el archivo: {str(e)}")
        return None

# Función para obtener los archivos del usuario
def get_user_files(user_id):
    """Obtiene la lista de archivos subidos por el usuario desde Firebase."""
    try:
        bucket = storage.bucket()
        # Obtener todos los blobs (archivos) en la carpeta del usuario
        blobs = bucket.list_blobs(prefix=f"users/{user_id}/")
        
        # Extraer solo los nombres de los archivos
        user_files = [blob.name.split('/')[-1] for blob in blobs if blob.name != f"users/{user_id}/"]
        return user_files
    except Exception as e:
        st.error(f"Error al obtener los archivos del usuario: {str(e)}")
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

def procesar_datos_sismicos(df, canales, corte_bajo, corte_alto, porcentaje_taper):
    fs_predeterminada = 100  # Puedes ajustar esto según tus datos o especificaciones del sensor
    fs = fs_predeterminada  # Inicializamos fs con el valor predeterminado
    
    # Verificar si existe una columna de tiempo
    columnas_tiempo = ['marca_tiempo', 'timestamp', 'tiempo']
    columna_tiempo = next((col for col in columnas_tiempo if col in df.columns), None)
    
    if columna_tiempo:
        try:
            # Intenta convertir la columna de tiempo a datetime
            df[columna_tiempo] = pd.to_datetime(df[columna_tiempo], errors='coerce')
            
            # Verifica si hay valores NaT (Not a Time) después de la conversión
            if df[columna_tiempo].isnull().any():
                st.warning(f"Algunos valores en la columna {columna_tiempo} no pudieron ser convertidos a fechas. Se usará un índice numérico en su lugar.")
                df[columna_tiempo] = pd.to_numeric(df.index)
            else:
                # Calcula la diferencia de tiempo si la conversión fue exitosa
                diferencia_tiempo = (df[columna_tiempo].iloc[1] - df[columna_tiempo].iloc[0]).total_seconds()
                if diferencia_tiempo > 0:
                    fs = 1 / diferencia_tiempo
        except Exception as e:
            st.warning(f"Error al procesar la columna de tiempo: {str(e)}. Se usará un índice numérico en su lugar.")
            df[columna_tiempo] = pd.to_numeric(df.index)
    else:
        # Si no hay columna de tiempo, usamos el índice como tiempo
        st.warning("No se encontró una columna de tiempo válida. Se usará el índice como tiempo.")
        df['tiempo'] = pd.to_numeric(df.index)
        columna_tiempo = 'tiempo'

    st.info(f"Frecuencia de muestreo utilizada: {fs} Hz")

    resultados = {}
    for canal in canales:
        # Aplicar filtro pasabanda
        datos_filtrados = aplicar_filtro_pasabanda(df[canal], corte_bajo=corte_bajo, corte_alto=corte_alto, fs=fs)
        
        # Aplicar taper
        datos_con_taper = aplicar_taper(datos_filtrados, porcentaje=porcentaje_taper)
        
        # Calcular FFT
        frecuencias, magnitudes = calcular_fft(datos_con_taper, fs)
        
        resultados[canal] = {
            'serie_tiempo': df[canal],
            'serie_filtrada': datos_con_taper,
            'frecuencias_fft': frecuencias,
            'magnitudes_fft': magnitudes
        }
    
    return resultados, fs, columna_tiempo

def graficar_resultados(resultados, fs, canales):
    if len(canales) == 1:
        canal = canales[0]
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            f"Canal {canal.upper()} (Original)",
            f"Canal {canal.upper()} (Filtrado)",
            f"Canal {canal.upper()} FFT",
            "Secciones seleccionadas para FFT"))

        tiempo = np.arange(len(resultados[canal]['serie_tiempo'])) / fs

        fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_tiempo'], name="Original"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_filtrada'], name="Filtrado"), row=1, col=2)
        
        # Centrar el gráfico FFT en los datos importantes
        frecuencias_fft = resultados[canal]['frecuencias_fft']
        magnitudes_fft = resultados[canal]['magnitudes_fft']
        indice_freq_max = np.argmax(magnitudes_fft)
        indice_inicio = max(0, indice_freq_max - 100)
        indice_fin = min(len(frecuencias_fft), indice_freq_max + 100)
        
        fig.add_trace(go.Scatter(x=frecuencias_fft[indice_inicio:indice_fin], 
                                 y=magnitudes_fft[indice_inicio:indice_fin], 
                                 name="FFT"), row=2, col=1)
    else:
        fig = make_subplots(rows=3, cols=1, subplot_titles=(
            "Canales Originales",
            "Canales Filtrados",
            "FFT de los Canales"))

        tiempo = np.arange(len(resultados['x']['serie_tiempo'])) / fs

        for i, canal in enumerate(canales):
            fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_tiempo'], name=f"{canal.upper()} Original"), row=1, col=1)
            fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_filtrada'], name=f"{canal.upper()} Filtrado"), row=2, col=1)
            fig.add_trace(go.Scatter(x=resultados[canal]['frecuencias_fft'], y=resultados[canal]['magnitudes_fft'], name=f"{canal.upper()} FFT"), row=3, col=1)

    fig.update_layout(height=1000, width=1000, title_text="Análisis de Canales")
    return fig

def copiar_espectro_al_portapapeles(resultados, canal):
    frecuencias = resultados[canal]['frecuencias_fft']
    magnitudes = resultados[canal]['magnitudes_fft']
    espectro_str = 'Frecuencia (Hz), Magnitud\n'
    espectro_str += '\n'.join([f'{freq},{mag}' for freq, mag in zip(frecuencias, magnitudes)])
    pyperclip.copy(espectro_str)
    st.success(f"Espectro de Fourier del canal {canal.upper()} copiado al portapapeles")

def seleccionar_secciones_aleatorias(datos, fs, num_secciones=5, duracion_seccion=30):
    longitud_seccion = int(duracion_seccion * fs)
    longitud_datos = len(datos)
    
    if longitud_datos <= longitud_seccion:
        st.warning("La duración de la sección es mayor o igual que los datos disponibles. Se utilizará toda la serie de datos.")
        return [(0, longitud_datos)]
    
    inicio_maximo = longitud_datos - longitud_seccion
    num_secciones_posibles = min(num_secciones, inicio_maximo)
    
    if num_secciones_posibles == 0:
        st.warning("No hay suficientes datos para seleccionar secciones. Se utilizará toda la serie de datos.")
        return [(0, longitud_datos)]
    
    if num_secciones_posibles < num_secciones:
        st.warning(f"Solo se pueden seleccionar {num_secciones_posibles} secciones con la duración especificada.")
    
    indices_inicio = random.sample(range(inicio_maximo + 1), num_secciones_posibles)
    return [(inicio, min(inicio + longitud_seccion, longitud_datos)) for inicio in indices_inicio]

def graficar_ratio_y_promedio(resultados, fs, tipo_ratio):
    numerador = 'x' if tipo_ratio == 'X/Z' else 'y'
    denominador = 'z'
    
    ratio = resultados[numerador]['magnitudes_fft'] / resultados[denominador]['magnitudes_fft']
    frecuencias = resultados[numerador]['frecuencias_fft']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frecuencias, y=ratio, name=f'Ratio {tipo_ratio}', line=dict(color='blue')))
    
    # Calcular y graficar el promedio móvil
    tamano_ventana = 10
    promedio_movil = np.convolve(ratio, np.ones(tamano_ventana)/tamano_ventana, mode='valid')
    fig.add_trace(go.Scatter(x=frecuencias[tamano_ventana-1:], y=promedio_movil, 
                             name=f'Promedio {tipo_ratio}', line=dict(color='red')))
    
    # Calcular y graficar la desviación estándar
    desviacion_estandar = np.std(ratio)
    fig.add_trace(go.Scatter(x=frecuencias, y=np.ones_like(frecuencias) * (np.mean(ratio) + desviacion_estandar),
                             name='Desviación Estándar Superior', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=frecuencias, y=np.ones_like(frecuencias) * (np.mean(ratio) - desviacion_estandar),
                             name='Desviación Estándar Inferior', line=dict(color='green', dash='dot')))
    
    fig.update_layout(title=f'Ratio {tipo_ratio}, Promedio y Desviación Estándar',
                      xaxis_title='Frecuencia (Hz)',
                      yaxis_title='Ratio',
                      height=600, width=1000)
    return fig

# Función principal de la aplicación
def main():
    st.title("Análisis del Acelerograma")

    # Autenticación
    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Iniciar sesión", "Registrarse"])
        
        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Contraseña", type="password")
            if st.button("Iniciar sesión"):
                user = sign_in(email, password)
                if user:
                    st.session_state.user = user
                    st.success("Sesión iniciada con éxito!")
                    st.experimental_rerun()

        with tab2:
            new_email = st.text_input("Nuevo Email")
            new_password = st.text_input("Nueva Contraseña", type="password")
            if st.button("Registrarse"):
                user = sign_up(new_email, new_password)
                if user:
                    st.session_state.user = user
                    st.success("Registrado con éxito!")
                    st.experimental_rerun()
    else:
        st.write(f"Bienvenido, {st.session_state.user.email}")
        if st.button("Cerrar sesión"):
            st.session_state.user = None
            st.experimental_rerun()

        # Subir archivo
        uploaded_file = st.file_uploader("Sube un archivo CSV o TXT", type=["csv", "txt"])
        if uploaded_file:
            file_url = upload_file(uploaded_file, st.session_state.user.uid)
            if file_url:
                st.success("Archivo subido exitosamente!")

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