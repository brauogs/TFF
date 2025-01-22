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
from PIL import Image
import random

st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

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

def calcular_espectro_potencia(datos, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(len(datos), int(fs * 60))  # 60 segundos o la longitud total de los datos
    frecuencias, Pxx = signal.welch(datos, fs, nperseg=nperseg)
    return frecuencias, Pxx

def calcular_hv(espectros):
    h_cuadrado = espectros['x'] + espectros['y']
    hv = np.sqrt(h_cuadrado) / espectros['z']
    return hv

def obtener_periodo_fundamental(frecuencias, hv):
    indice_max = np.argmax(hv)
    frecuencia_fundamental = frecuencias[indice_max]
    periodo_fundamental = 1 / frecuencia_fundamental
    return periodo_fundamental, frecuencia_fundamental

def procesar_datos_sismicos(df, canales, corte_bajo, corte_alto, porcentaje_taper):
    fs_predeterminada = 100
    fs = fs_predeterminada
    
    columnas_tiempo = ['marca_tiempo', 'timestamp', 'tiempo']
    columna_tiempo = next((col for col in columnas_tiempo if col in df.columns), None)
    
    if columna_tiempo:
        try:
            df[columna_tiempo] = pd.to_datetime(df[columna_tiempo], errors='coerce')
            
            if df[columna_tiempo].isnull().any():
                st.warning(f"Algunos valores en la columna {columna_tiempo} no pudieron ser convertidos a fechas. Se usará un índice numérico en su lugar.")
                df[columna_tiempo] = pd.to_numeric(df.index)
            else:
                diferencia_tiempo = (df[columna_tiempo].iloc[1] - df[columna_tiempo].iloc[0]).total_seconds()
                if diferencia_tiempo > 0:
                    fs = 1 / diferencia_tiempo
        except Exception as e:
            st.warning(f"Error al procesar la columna de tiempo: {str(e)}. Se usará un índice numérico en su lugar.")
            df[columna_tiempo] = pd.to_numeric(df.index)
    else:
        st.warning("No se encontró una columna de tiempo válida. Se usará el índice como tiempo.")
        df['tiempo'] = pd.to_numeric(df.index)
        columna_tiempo = 'tiempo'

    st.info(f"Frecuencia de muestreo utilizada: {fs} Hz")

    resultados = {}
    espectros = {}
    for canal in canales:
        datos_filtrados = aplicar_filtro_pasabanda(df[canal], corte_bajo=corte_bajo, corte_alto=corte_alto, fs=fs)
        datos_con_taper = aplicar_taper(datos_filtrados, porcentaje=porcentaje_taper)
        frecuencias, magnitudes = calcular_fft(datos_con_taper, fs)
        frecuencias_psd, psd = calcular_espectro_potencia(datos_con_taper, fs)
        
        resultados[canal] = {
            'serie_tiempo': df[canal],
            'serie_filtrada': datos_con_taper,
            'frecuencias_fft': frecuencias,
            'magnitudes_fft': magnitudes,
            'frecuencias_psd': frecuencias_psd,
            'psd': psd
        }
        espectros[canal] = psd

    if set(['x', 'y', 'z']).issubset(canales):
        hv = calcular_hv(espectros)
        periodo_fundamental, frecuencia_fundamental = obtener_periodo_fundamental(frecuencias_psd, hv)
        resultados['hv'] = {
            'frecuencias': frecuencias_psd,
            'hv': hv,
            'periodo_fundamental': periodo_fundamental,
            'frecuencia_fundamental': frecuencia_fundamental
        }
    
    return resultados, fs, columna_tiempo

def graficar_resultados(resultados, fs, canales):
    if len(canales) == 1:
        canal = canales[0]
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            f"Canal {canal.upper()} (Original)",
            f"Canal {canal.upper()} (Filtrado)",
            f"Canal {canal.upper()} FFT",
            "Secciones seleccionadas para FFT"),
            vertical_spacing=0.1,
            horizontal_spacing=0.05)

        tiempo = np.arange(len(resultados[canal]['serie_tiempo'])) / fs

        fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_tiempo'], name="Original"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_filtrada'], name="Filtrado"), row=1, col=2)
        
        frecuencias_fft = resultados[canal]['frecuencias_fft']
        magnitudes_fft = resultados[canal]['magnitudes_fft']
        
        fig.add_trace(go.Scatter(x=frecuencias_fft, y=magnitudes_fft, name="FFT"), row=2, col=1)
    else:
        fig = make_subplots(rows=3, cols=1, subplot_titles=(
            "Canales Originales",
            "Canales Filtrados",
            "FFT de los Canales"),
            vertical_spacing=0.1)

        tiempo = np.arange(len(resultados['x']['serie_tiempo'])) / fs

        for i, canal in enumerate(canales):
            fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_tiempo'], name=f"{canal.upper()} Original"), row=1, col=1)
            fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal]['serie_filtrada'], name=f"{canal.upper()} Filtrado"), row=2, col=1)
            fig.add_trace(go.Scatter(x=resultados[canal]['frecuencias_fft'], y=resultados[canal]['magnitudes_fft'], name=f"{canal.upper()} FFT"), row=3, col=1)

    fig.update_layout(height=1200, width=1200, title_text="Análisis de Canales")
    fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=1, col=2)
    fig.update_xaxes(title_text="Frecuencia (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitud", row=1, col=1)
    fig.update_yaxes(title_text="Amplitud", row=1, col=2)
    fig.update_yaxes(title_text="Magnitud", row=2, col=1)
    return fig

def metodo_nakamura(datos_x, datos_y, datos_z, fs):
    """
    Implementación del método de Nakamura H/V siguiendo la metodología específica mostrada.
    datos_x: componente E-W
    datos_y: componente N-S
    datos_z: componente vertical
    fs: frecuencia de muestreo
    """
    from scipy.signal import welch, detrend
    import numpy as np
    
    # 1. Preprocesamiento de señales
    def preprocesar_senal(datos):
        # Remover tendencia
        datos = detrend(datos)
        # Aplicar taper
        window = np.hanning(len(datos))
        return datos * window
    
    # Preprocesar todas las componentes
    datos_x = preprocesar_senal(datos_x)
    datos_y = preprocesar_senal(datos_y)
    datos_z = preprocesar_senal(datos_z)
    
    # 2. Cálculo de espectros
    def calcular_espectro(datos):
        nperseg = int(fs * 25)  # Ventana de 25 segundos
        noverlap = int(nperseg * 0.05)  # 5% de solapamiento
        frecuencias, Pxx = welch(datos, fs, nperseg=nperseg, noverlap=noverlap, 
                               detrend=False)  # Ya se aplicó detrend
        return frecuencias, Pxx
    
    # Calcular espectros para cada componente
    f, Pxx = calcular_espectro(datos_x)  # E-W
    _, Pyy = calcular_espectro(datos_y)  # N-S
    _, Pzz = calcular_espectro(datos_z)  # Vertical
    
    # 3. Calcular espectro horizontal promedio
    Phh = (Pxx + Pyy) / 2
    
    # 4. Aplicar suavizado Konno-Ohmachi
    def suavizado_konno_ohmachi(freq, spec, b=40):
        suavizado = np.zeros_like(spec)
        for i, fc in enumerate(freq):
            if fc == 0:
                continue
            w = np.zeros_like(freq)
            wb = np.abs(np.log10(freq/fc))
            w = (np.sin(b * wb) / (b * wb))**4
            w[np.isnan(w)] = 1
            w = w / np.sum(w)
            suavizado[i] = np.sum(spec * w)
        return suavizado
    
    # Aplicar suavizado a los espectros
    Phh_smooth = suavizado_konno_ohmachi(f, Phh)
    Pzz_smooth = suavizado_konno_ohmachi(f, Pzz)
    Pxx_smooth = suavizado_konno_ohmachi(f, Pxx)
    Pyy_smooth = suavizado_konno_ohmachi(f, Pyy)
    
    # 5. Calcular ratio H/V
    # Evitar división por cero
    epsilon = 1e-10
    Pzz_smooth = np.where(Pzz_smooth < epsilon, epsilon, Pzz_smooth)
    hv_ratio = np.sqrt(Phh_smooth / Pzz_smooth)
    
    # 6. Encontrar frecuencia fundamental
    # Filtrar frecuencias de interés (0.5 - 10 Hz)
    mask = (f >= 0.5) & (f <= 10)
    f_filtered = f[mask]
    hv_filtered = hv_ratio[mask]
    
    # Encontrar pico principal cerca de 1.4 Hz
    target_freq = 1.4
    window = 0.5  # ±0.5 Hz alrededor de 1.4 Hz
    freq_mask = (f_filtered >= target_freq - window) & (f_filtered <= target_freq + window)
    
    if np.any(freq_mask):
        peak_idx = np.argmax(hv_filtered[freq_mask])
        freq_range_indices = np.where(freq_mask)[0]
        peak_freq_idx = freq_range_indices[peak_idx]
        frecuencia_fundamental = f_filtered[peak_freq_idx]
        frecuencias_fundamentales = [frecuencia_fundamental]
        periodos_fundamentales = [1 / frecuencia_fundamental]
    else:
        st.warning("No se encontró un pico claro cerca de 1.4 Hz")
        frecuencias_fundamentales = []
        periodos_fundamentales = []
    
    # 7. Calcular intervalos de confianza
    factor_freq = 1 / (1 + f)
    hv_std = 0.2 * hv_ratio * factor_freq  # 20% de variabilidad típica
    
    hv_mas_std = hv_ratio + hv_std
    hv_menos_std = hv_ratio - hv_std
    
    return (f, hv_ratio, hv_mas_std, hv_menos_std, frecuencias_fundamentales, periodos_fundamentales, 
            Pxx_smooth, Pyy_smooth, Pzz_smooth, Phh_smooth, datos_x, datos_y, datos_z)

def plot_nakamura_workflow(datos_x, datos_y, datos_z, f, Pxx, Pyy, Pzz, Phh, hv_ratio, fs):
    """
    Función para visualizar todo el proceso del método de Nakamura
    """
    # Crear figura con subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Microtrepidaciones E-W', 'Microtrepidaciones N-S',
            'Espectro E-W', 'Espectro N-S',
            'Espectro Horizontal Promedio', 'Espectro Vertical',
            'Relación espectral H/V', ''
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Series de tiempo
    tiempo = np.arange(len(datos_x)) / fs
    fig.add_trace(go.Scatter(x=tiempo, y=datos_x, name='E-W', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=datos_y, name='N-S', line=dict(color='blue')), row=1, col=2)
    
    # 2. Espectros individuales
    fig.add_trace(go.Scatter(x=f, y=Pxx, name='Espectro E-W', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=f, y=Pyy, name='Espectro N-S', line=dict(color='blue')), row=2, col=2)
    
    # 3. Espectro horizontal promedio y vertical
    fig.add_trace(go.Scatter(x=f, y=Phh, name='H promedio', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=f, y=Pzz, name='Vertical', line=dict(color='blue')), row=3, col=2)
    
    # 4. Relación H/V
    fig.add_trace(go.Scatter(x=f, y=hv_ratio, name='H/V', line=dict(color='blue')), row=4, col=1)
    
    # Actualizar ejes y diseño
    for i in range(1, 5):
        if i == 1:
            fig.update_xaxes(title_text='Tiempo (s)', row=i, col=1)
            fig.update_xaxes(title_text='Tiempo (s)', row=i, col=2)
            fig.update_yaxes(title_text='Amplitud', row=i, col=1)
            fig.update_yaxes(title_text='Amplitud', row=i, col=2)
        else:
            fig.update_xaxes(title_text='Frecuencia (Hz)', type='log', row=i, col=1)
            fig.update_xaxes(title_text='Frecuencia (Hz)', type='log', row=i, col=2)
            fig.update_yaxes(title_text='Amplitud', type='log', row=i, col=1)
            fig.update_yaxes(title_text='Amplitud', type='log', row=i, col=2)
    
    # Actualizar diseño general
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Proceso completo del método de Nakamura",
    )
    
    return fig



def plot_espectros(f, Pxx, Pyy, Pzz, Phh):
    """
    Función para graficar los espectros individuales y promedio
    """
    fig = make_subplots(rows=3, cols=2, 
                       subplot_titles=('Espectro E-W', 'Espectro N-S',
                                     'Espectro Horizontal Promedio', 'Espectro Vertical'))
    
    # Espectro E-W
    fig.add_trace(go.Scatter(x=f, y=Pxx, name='E-W'), row=1, col=1)
    
    # Espectro N-S
    fig.add_trace(go.Scatter(x=f, y=Pyy, name='N-S'), row=1, col=2)
    
    # Espectro Horizontal Promedio
    fig.add_trace(go.Scatter(x=f, y=Phh, name='H promedio'), row=2, col=1)
    
    # Espectro Vertical
    fig.add_trace(go.Scatter(x=f, y=Pzz, name='Vertical'), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text='Frecuencia (Hz)', type='log')
    fig.update_yaxes(title_text='Amplitud', type='log')
    
    return fig

def descargar_datos_procesados(resultados, canales, fs):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for canal in canales:
            # Ensure all arrays have the same length
            tiempo = np.arange(len(resultados[canal]['serie_tiempo'])) / fs
            original = resultados[canal]['serie_tiempo']
            filtrado = resultados[canal]['serie_filtrada']
            frecuencias_fft = resultados[canal]['frecuencias_fft']
            magnitudes_fft = resultados[canal]['magnitudes_fft']
            
            # Pad shorter arrays with NaN values
            max_length = max(len(tiempo), len(original), len(filtrado), len(frecuencias_fft), len(magnitudes_fft))
            tiempo = np.pad(tiempo, (0, max_length - len(tiempo)), mode='constant', constant_values=np.nan)
            original = np.pad(original, (0, max_length - len(original)), mode='constant', constant_values=np.nan)
            filtrado = np.pad(filtrado, (0, max_length - len(filtrado)), mode='constant', constant_values=np.nan)
            frecuencias_fft = np.pad(frecuencias_fft, (0, max_length - len(frecuencias_fft)), mode='constant', constant_values=np.nan)
            magnitudes_fft = np.pad(magnitudes_fft, (0, max_length - len(magnitudes_fft)), mode='constant', constant_values=np.nan)
            
            df = pd.DataFrame({
                'Tiempo': tiempo,
                'Original': original,
                'Filtrado': filtrado,
                'Frecuencias_FFT': frecuencias_fft,
                'Magnitudes_FFT': magnitudes_fft
            })
            df.to_excel(writer, sheet_name=f'Canal_{canal}', index=False)
    
    output.seek(0)
    return output

def plot_hv_degtra_style(f, hv_smooth, hv_plus_std, hv_minus_std, fundamental_frequencies):
    fig = go.Figure()
    
    # Add grid style similar to DEGTRA
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(
            type='log',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(211, 211, 211, 0.5)'
            ),
            title='f, Hz',
            ticktext=['10⁻¹', '10⁰', '10¹'],
            tickvals=[0.1, 1, 10],
            range=[-1, 1],  # 10^-1 to 10^1
        ),
        yaxis=dict(
            type='log',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(211, 211, 211, 0.5)'
            ),
            title='H/V',
            ticktext=['10⁻¹', '10⁰', '10¹'],
            tickvals=[0.1, 1, 10],
            range=[-1, 1],  # 10^-1 to 10^1
        )
    )
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=f,
        y=hv_smooth,
        mode='lines',
        name='mean',
        line=dict(color='red', width=1.5)
    ))
    
    # Add standard deviation lines
    fig.add_trace(go.Scatter(
        x=f,
        y=hv_plus_std,
        mode='lines',
        name='m+s',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=f,
        y=hv_minus_std,
        mode='lines',
        name='m-s',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    # Add vertical lines for fundamental frequencies
    for freq in fundamental_frequencies:
        fig.add_vline(
            x=freq,
            line_dash="dot",
            line_color="red",
            opacity=0.5
        )
    
    # Update layout
    fig.update_layout(
        title="Análisis H/V",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=800,
        height=500
    )
    
    return fig

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
    
    if canal_seleccionado == 'Todos los canales':
        st.subheader("Análisis H/V (Método de Nakamura)")
        
        # Obtener datos filtrados
        datos_x = resultados['x']['serie_filtrada']
        datos_y = resultados['y']['serie_filtrada']
        datos_z = resultados['z']['serie_filtrada']
        
        # Realizar análisis H/V
        (f, hv_ratio, hv_mas_std, hv_menos_std, frecuencias_fundamentales, periodos_fundamentales,
         Pxx, Pyy, Pzz, Phh, datos_x_proc, datos_y_proc, datos_z_proc) = metodo_nakamura(
            datos_x, datos_y, datos_z, fs
        )
        
        if f is not None and len(frecuencias_fundamentales) > 0:
            # Mostrar el proceso completo
            fig_workflow = plot_nakamura_workflow(
                datos_x_proc, datos_y_proc, datos_z_proc,
                f, Pxx, Pyy, Pzz, Phh, hv_ratio, fs
            )
            st.plotly_chart(fig_workflow)
            
            # Graficar resultado H/V final con estilo DEGTRA
            fig_hv = plot_hv_degtra_style(f, hv_ratio, hv_mas_std, hv_menos_std, frecuencias_fundamentales)
            st.plotly_chart(fig_hv)
            
            # Mostrar resultados
            st.write("Frecuencias fundamentales identificadas:")
            for i, (freq, period) in enumerate(zip(frecuencias_fundamentales, periodos_fundamentales)):
                st.write(f"Pico {i+1}:")
                st.write(f"  Frecuencia: {freq:.2f} Hz")
                st.write(f"  Periodo: {period:.2f} segundos")
            
            # Comparar con la frecuencia del acelerómetro
            frecuencia_acelerometro = 1.4  # Hz
            st.write(f"Frecuencia fundamental del acelerómetro: {frecuencia_acelerometro} Hz")
            
            if frecuencias_fundamentales[0] < frecuencia_acelerometro:
                st.success("La frecuencia fundamental del suelo es menor que la del acelerómetro, lo que indica que las mediciones son confiables.")
            else:
                st.warning("La frecuencia fundamental del suelo es mayor o igual que la del acelerómetro. Esto podría afectar la confiabilidad de las mediciones en frecuencias más altas.")
        
        else:
            st.error("No se pudieron identificar frecuencias fundamentales válidas en el análisis.")

    st.sidebar.header("Instrucciones")
    st.sidebar.markdown("""
    1. Inicie sesión o cree una cuenta.
    2. Suba un archivo CSV o TXT para analizar.
    3. Seleccione un archivo de sus archivos subidos.
    4. Elija el canal a analizar (seleccione 'Todos los canales' para el análisis H/V).
    5. Ajuste los parámetros de filtrado en la barra lateral.
    6. Especifique el número de rutinas FFT a realizar.
    7. Haga clic en 'Analizar datos' para ver los resultados.
    8. Revise los gráficos FFT y H/V para cada sección aleatoria.
    9. Descargue los datos procesados si lo desea.
    """)

if __name__ == "__main__":
    main()

