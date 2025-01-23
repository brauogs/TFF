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

def procesar_datos_sismicos(df, canales, corte_bajo, corte_alto):
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
    for canal in canales:
        datos = df[canal].values
        datos_corregidos = corregir_linea_base(datos)
        datos_filtrados = aplicar_filtro_pasabanda(datos_corregidos, corte_bajo=corte_bajo, corte_alto=corte_alto, fs=fs)
        resultados[canal] = datos_filtrados
    
    return resultados, fs, columna_tiempo

def corregir_linea_base(datos):
    return datos - np.mean(datos)

def calcular_espectro_fourier(datos, fs):
    n = len(datos)
    frecuencias = fftfreq(n, 1/fs)[:n//2]
    amplitudes = np.abs(fft(datos))[:n//2] * 2 / n
    return frecuencias, amplitudes

def analisis_hv(x, y, z, fs, num_ventanas=20, tamano_ventana=2000):
    cociente_xz = np.zeros(tamano_ventana // 2)
    cociente_yz = np.zeros(tamano_ventana // 2)
    cociente_xz2 = np.zeros(tamano_ventana // 2)
    cociente_yz2 = np.zeros(tamano_ventana // 2)
    
    for _ in range(num_ventanas):
        nini = random.randint(0, len(x) - tamano_ventana)
        x1 = x[nini:nini+tamano_ventana]
        y1 = y[nini:nini+tamano_ventana]
        z1 = z[nini:nini+tamano_ventana]
        
        fx, ax = calcular_espectro_fourier(x1, fs)
        _, ay = calcular_espectro_fourier(y1, fs)
        _, az = calcular_espectro_fourier(z1, fs)
        
        cociente_xz += ax / az / num_ventanas
        cociente_yz += ay / az / num_ventanas
        cociente_xz2 += (ax / az)**2 / num_ventanas
        cociente_yz2 += (ay / az)**2 / num_ventanas
    
    var_xz = cociente_xz2 - cociente_xz**2
    var_yz = cociente_yz2 - cociente_yz**2
    
    std_xz = np.sqrt(var_xz)
    std_yz = np.sqrt(var_yz)
    
    return fx, cociente_xz, cociente_yz, std_xz, std_yz

def graficar_resultados(resultados, fs, canales):
    fig = make_subplots(rows=len(canales), cols=2, 
                        subplot_titles=[f"Canal {canal.upper()} (Filtrado)" for canal in canales] + 
                                       [f"Canal {canal.upper()} FFT" for canal in canales],
                        vertical_spacing=0.1)

    for i, canal in enumerate(canales, start=1):
        tiempo = np.arange(len(resultados[canal])) / fs
        fig.add_trace(go.Scatter(x=tiempo, y=resultados[canal], name=f"{canal.upper()} Filtrado"), row=i, col=1)
        
        frecuencias, amplitudes = calcular_espectro_fourier(resultados[canal], fs)
        fig.add_trace(go.Scatter(x=frecuencias, y=amplitudes, name=f"{canal.upper()} FFT"), row=i, col=2)

    fig.update_layout(height=300*len(canales), width=1200, title_text="Análisis de Canales")
    for i in range(1, len(canales)+1):
        fig.update_xaxes(title_text="Tiempo (s)", row=i, col=1)
        fig.update_xaxes(title_text="Frecuencia (Hz)", row=i, col=2)
        fig.update_yaxes(title_text="Amplitud", row=i, col=1)
        fig.update_yaxes(title_text="Magnitud", row=i, col=2)
    return fig

def metodo_nakamura(datos_x, datos_y, datos_z, fs):
    from scipy.signal import savgol_filter
    from scipy.stats import chi2
    
    # Calculate power spectral density for each component
    f, Pxx = signal.welch(datos_x, fs, nperseg=min(len(datos_x), int(fs * 60)))
    _, Pyy = signal.welch(datos_y, fs, nperseg=min(len(datos_y), int(fs * 60)))
    _, Pzz = signal.welch(datos_z, fs, nperseg=min(len(datos_z), int(fs * 60)))
    
    # Calculate H/V ratio
    hv = np.sqrt((Pxx + Pyy) / (2 * Pzz))
    
    # Smooth H/V curve
    hv_smooth = savgol_filter(hv, window_length=11, polyorder=3)
    
    # Calculate confidence intervals
    df = 4  # Degrees of freedom (can be adjusted)
    ci_low = hv_smooth * chi2.ppf(0.05, df) / df
    ci_high = hv_smooth * chi2.ppf(0.95, df) / df
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hv_smooth, height=np.mean(hv_smooth), distance=int(len(f)/10))
    
    # Sort peaks by amplitude
    peak_amplitudes = hv_smooth[peaks]
    sorted_peak_indices = np.argsort(peak_amplitudes)[::-1]
    sorted_peaks = peaks[sorted_peak_indices]
    
    # Get the top 3 peaks
    top_peaks = sorted_peaks[:3]
    fundamental_frequencies = f[top_peaks]
    fundamental_periods = 1 / fundamental_frequencies
    
    # SESAME criteria
    def check_sesame_criteria(f0, hv_f0):
        criteria_met = []
        
        # Criterion 1: f0 > 10 / window_length
        window_length = len(datos_x) / fs
        criteria_met.append(f0 > 10 / window_length)
        
        # Criterion 2: nc = window_length * f0 > 200
        nc = window_length * f0
        criteria_met.append(nc > 200)
        
        # Criterion 3: σA(f) < 2 for 0.5f0 < f < 2f0 when f0 > 0.5 Hz
        #              σA(f) < 3 for 0.5f0 < f < 2f0 when f0 < 0.5 Hz
        sigma_a = np.std(hv_smooth[(f > 0.5*f0) & (f < 2*f0)])
        criteria_met.append(sigma_a < (3 if f0 < 0.5 else 2))
        
        # Criterion 4: σf < ε(f0) where ε(f0) is defined in SESAME guidelines
        sigma_f = np.std(f[(f > 0.5*f0) & (f < 2*f0)])
        epsilon = 0.25 * f0 if f0 > 1 else 0.25 * f0 ** 1.78
        criteria_met.append(sigma_f < epsilon)
        
        # Criterion 5: Amplitude of the peak > 2
        criteria_met.append(hv_f0 > 2)
        
        return criteria_met
    
    sesame_results = [check_sesame_criteria(ff, hv_smooth[top_peaks[i]]) for i, ff in enumerate(fundamental_frequencies)]
    
    return f, hv_smooth, ci_low, ci_high, fundamental_frequencies, fundamental_periods, sesame_results

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
            tiempo = np.arange(len(resultados[canal])) / fs
            original = resultados[canal] #Simplified
            
            # Pad shorter arrays with NaN values
            max_length = max(len(tiempo), len(original))
            tiempo = np.pad(tiempo, (0, max_length - len(tiempo)), mode='constant', constant_values=np.nan)
            original = np.pad(original, (0, max_length - len(original)), mode='constant', constant_values=np.nan)
            
            df = pd.DataFrame({
                'Tiempo': tiempo,
                'Filtrado': original #Simplified
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

            st.sidebar.header("Parámetros de filtrado")
            corte_bajo = st.sidebar.slider("Fmin (Hz)", 0.01, 1.0, 0.05, 0.01)
            corte_alto = st.sidebar.slider("Fmax (Hz)", 1.0, 50.0, 10.0, 0.1)

            num_ventanas = st.sidebar.number_input("Número de ventanas para análisis H/V", min_value=1, max_value=100, value=20)

            if st.button("Analizar datos"):
                canales = ['x', 'y', 'z']
                resultados, fs, columna_tiempo = procesar_datos_sismicos(df, canales, corte_bajo, corte_alto)
                fig = graficar_resultados(resultados, fs, canales)
                st.plotly_chart(fig)

                st.subheader("Análisis H/V")
                frecuencias_hv, hv_x_promedio, hv_y_promedio, hv_x_std, hv_y_std = analisis_hv(resultados['x'], resultados['y'], resultados['z'], fs, num_ventanas=num_ventanas)
    
                fig_hv = go.Figure()
                fig_hv.add_trace(go.Scatter(x=frecuencias_hv, y=hv_x_promedio, name="H/V (X/Z) Promedio"))
                fig_hv.add_trace(go.Scatter(x=frecuencias_hv, y=hv_x_promedio + hv_x_std, name="H/V (X/Z) +Std", line=dict(dash='dash')))
                fig_hv.add_trace(go.Scatter(x=frecuencias_hv, y=hv_x_promedio - hv_x_std, name="H/V (X/Z) -Std", line=dict(dash='dash')))
                fig_hv.add_trace(go.Scatter(x=frecuencias_hv, y=hv_y_promedio, name="H/V (Y/Z) Promedio"))
                fig_hv.add_trace(go.Scatter(x=frecuencias_hv, y=hv_y_promedio + hv_y_std, name="H/V (Y/Z) +Std", line=dict(dash='dash')))
                fig_hv.add_trace(go.Scatter(x=frecuencias_hv, y=hv_y_promedio - hv_y_std, name="H/V (Y/Z) -Std", line=dict(dash='dash')))
                
                fig_hv.update_layout(
                    title="Análisis H/V",
                    xaxis_title="Frecuencia (Hz)",
                    yaxis_title="Ratio H/V",
                    xaxis_type="log"
                )
                st.plotly_chart(fig_hv)

                st.write("Resultados del análisis H/V:")
                st.write(f"Promedio H/V (X/Z): {np.mean(hv_x_promedio):.2f}")
                st.write(f"Promedio H/V (Y/Z): {np.mean(hv_y_promedio):.2f}")
                st.write(f"Desviación estándar promedio H/V (X/Z): {np.mean(hv_x_std):.2f}")
                st.write(f"Desviación estándar promedio H/V (Y/Z): {np.mean(hv_y_std):.2f}")

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
    4. Ajuste los parámetros de filtrado en la barra lateral.
    5. Especifique el número de ventanas para el análisis H/V.
    6. Haga clic en 'Analizar datos' para ver los resultados.
    7. Revise los gráficos de los canales filtrados, FFT y análisis H/V.
    """)

if __name__ == "__main__":
    main()

