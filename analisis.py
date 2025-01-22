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
    # Asegurar que las señales tengan la misma longitud
    min_length = min(len(datos_x), len(datos_y), len(datos_z))
    datos_x = datos_x[:min_length]
    datos_y = datos_y[:min_length]
    datos_z = datos_z[:min_length]
    
    # Definir parámetros de ventana
    window_length = int(fs * 25)  # Ventanas de 25 segundos
    overlap = 0.5  # 50% de solapamiento
    step = int(window_length * (1 - overlap))
    n_windows = (min_length - window_length) // step + 1

    hv_ratios = []
    frequencies = None

    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_length

        # Ventanas sincronizadas
        window_x = datos_x[start_idx:end_idx]
        window_y = datos_y[start_idx:end_idx]
        window_z = datos_z[start_idx:end_idx]

        # Aplicar taper
        taper = np.hanning(window_length)
        window_x *= taper
        window_y *= taper
        window_z *= taper

        # Espectros de potencia
        f, Pxx = welch(window_x, fs, nperseg=window_length, noverlap=None, detrend='constant')
        _, Pyy = welch(window_y, fs, nperseg=window_length, noverlap=None, detrend='constant')
        _, Pzz = welch(window_z, fs, nperseg=window_length, noverlap=None, detrend='constant')

        # Relación H/V
        hv = np.sqrt((Pxx + Pyy) / (2 * Pzz))
        hv_ratios.append(hv)
        
        if frequencies is None:
            frequencies = f

    # Calcular media y desviación estándar
    hv_ratios = np.array(hv_ratios)
    hv_mean = np.mean(hv_ratios, axis=0)
    hv_std = np.std(hv_ratios, axis=0)

    # Suavizado Konno-Ohmachi
    hv_smooth = konno_ohmachi_smooth(frequencies, hv_mean)

    # Ajustar desviaciones estándar
    freq_factor = 1 / (1 + frequencies)
    hv_std_adjusted = hv_std * freq_factor
    hv_plus_std = hv_smooth + hv_std_adjusted
    hv_minus_std = hv_smooth - hv_std_adjusted

    # Detectar picos
    peaks, _ = find_peaks(hv_smooth, height=np.mean(hv_smooth), distance=int(len(frequencies)/10))
    peak_amplitudes = hv_smooth[peaks]
    sorted_peaks = peaks[np.argsort(peak_amplitudes)[::-1]]
    fundamental_frequencies = frequencies[sorted_peaks]

    # Filtrar picos por rango esperado
    expected_range = (fundamental_frequencies >= 0.5) & (fundamental_frequencies <= 2.0)
    fundamental_frequencies = fundamental_frequencies[expected_range]
    fundamental_periods = 1 / fundamental_frequencies

    return frequencies, hv_smooth, hv_plus_std, hv_minus_std, fundamental_frequencies, fundamental_periods

def seleccionar_secciones_aleatorias(datos, fs, num_secciones=5, duracion_seccion=30):
    longitud_seccion = int(duracion_seccion * fs)
    longitud_datos = len(datos)
    
    if longitud_datos <= longitud_seccion:
        st.warning("La duración de la sección es mayor o igual que los datos disponibles. Se utilizará toda la serie de datos.")
        return [(0, longitud_datos, datos)]
    
    inicio_maximo = longitud_datos - longitud_seccion
    num_secciones_posibles = min(num_secciones, inicio_maximo)
    
    if num_secciones_posibles == 0:
        st.warning("No hay suficientes datos para seleccionar secciones. Se utilizará toda la serie de datos.")
        return [(0, longitud_datos, datos)]
    
    if num_secciones_posibles < num_secciones:
        st.warning(f"Solo se pueden seleccionar {num_secciones_posibles} secciones con la duración especificada.")
    
    indices_inicio = random.sample(range(inicio_maximo + 1), num_secciones_posibles)
    return [(inicio, min(inicio + longitud_seccion, longitud_datos), datos[inicio:min(inicio + longitud_seccion, longitud_datos)]) for inicio in indices_inicio]

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
            range=[-1, 2]  # 10^-1 to 10^2
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
            range=[-1, 1]  # 10^-1 to 10^1
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
        title="Análisis H/V (Estilo DEGTRA)",
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
                fig = graficar_resultados(resultados, fs, canales)
                st.plotly_chart(fig)

                st.subheader("Rutinas FFT y Análisis H/V (Método de Nakamura) en secciones aleatorias")
        
                if canal_seleccionado == 'Todos los canales':
                    secciones_x = seleccionar_secciones_aleatorias(resultados['x']['serie_filtrada'], fs, num_secciones=num_rutinas_fft)
                    secciones_y = seleccionar_secciones_aleatorias(resultados['y']['serie_filtrada'], fs, num_secciones=num_rutinas_fft)
                    secciones_z = seleccionar_secciones_aleatorias(resultados['z']['serie_filtrada'], fs, num_secciones=num_rutinas_fft)

                    frecuencias_fundamentales = []
                    periodos_fundamentales = []

                    for i, ((inicio_x, fin_x, datos_x), (_, _, datos_y), (_, _, datos_z)) in enumerate(zip(secciones_x, secciones_y, secciones_z)):
                        st.write(f"Sección {i+1}")
                        
                        # FFT analysis
                        fig_fft = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                        for j, (canal, datos) in enumerate([('X', datos_x), ('Y', datos_y), ('Z', datos_z)]):
                            frecuencias, magnitudes = calcular_fft(datos, fs)
                            fig_fft.add_trace(go.Scatter(x=frecuencias, y=magnitudes, name=f"FFT Canal {canal}"), row=j+1, col=1)
                            fig_fft.update_yaxes(title_text=f"Magnitud {canal}", row=j+1, col=1)
                        
                        fig_fft.update_layout(height=800, title_text=f"FFT Sección {i+1}")
                        fig_fft.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1)
                        st.plotly_chart(fig_fft)

                        # H/V analysis using improved Nakamura method
                        f, hv_smooth, hv_plus_std, hv_minus_std, fundamental_frequencies_section, fundamental_periods_section = metodo_nakamura(datos_x, datos_y, datos_z, fs)

                        fig_hv = plot_hv_degtra_style(f, hv_smooth, hv_plus_std, hv_minus_std, fundamental_frequencies_section)
                        st.plotly_chart(fig_hv)

                        st.write("Frecuencias fundamentales identificadas:")
                        for k, (freq, period) in enumerate(zip(fundamental_frequencies_section, fundamental_periods_section)):
                            st.write(f"Pico {k+1}:")
                            st.write(f"  Frecuencia: {freq:.2f} Hz")
                            st.write(f"  Periodo: {period:.2f} segundos")


                        frecuencias_fundamentales.append(fundamental_frequencies_section)
                        periodos_fundamentales.append(fundamental_periods_section)

                    # Calculate and display average fundamental frequency and period
                    st.subheader("Resumen de resultados")
                    for i in range(min(3, len(frecuencias_fundamentales[0]))):
                        promedio_frecuencia = np.mean([f[i] for f in frecuencias_fundamentales if len(f) > i])
                        promedio_periodo = np.mean([p[i] for p in periodos_fundamentales if len(p) > i])
                        st.write(f"Pico {i+1}:")
                        st.write(f"  Promedio de frecuencias fundamentales: {promedio_frecuencia:.2f} Hz")
                        st.write(f"  Promedio de periodos fundamentales: {promedio_periodo:.2f} segundos")

                    # Compare with accelerometer's fundamental frequency
                    frecuencia_acelerometro = 1.4  # Hz
                    st.write(f"Frecuencia fundamental del acelerómetro: {frecuencia_acelerometro} Hz")

                    if all(np.mean([f[0] for f in frecuencias_fundamentales]) < frecuencia_acelerometro for f in frecuencias_fundamentales):
                        st.success("La frecuencia fundamental promedio del suelo (primer pico) es menor que la del acelerómetro, lo que indica que las mediciones son confiables.")
                    else:
                        st.warning("La frecuencia fundamental promedio del suelo (primer pico) es mayor o igual que la del acelerómetro. Esto podría afectar la confiabilidad de las mediciones en frecuencias más altas.")

                    st.info("Nota: Se recomienda revisar los criterios SESAME para cada sección y pico para evaluar la confiabilidad de los resultados.")

                else:
                    st.warning("El análisis H/V requiere datos de los tres canales (X, Y, Z). Por favor, seleccione 'Todos los canales' para realizar este análisis.")

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
    4. Elija el canal a analizar (seleccione 'Todos los canales' para el análisis H/V).
    5. Ajuste los parámetros de filtrado en la barra lateral.
    6. Especifique el número de rutinas FFT a realizar.
    7. Haga clic en 'Analizar datos' para ver los resultados.
    8. Revise los gráficos FFT y H/V para cada sección aleatoria.
    9. Descargue los datos procesados si lo desea.
    """)

if __name__ == "__main__":
    main()

