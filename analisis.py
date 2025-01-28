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

# Configuración de Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["firebase"])
    firebase_admin.initialize_app(cred, {'storageBucket': "vibraciones-aac24.appspot.com"})

def registrar(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user
    except Exception as e:
        st.error(f"Error durante el registro: {str(e)}")
        return None

def iniciar_sesion(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user
    except Exception as e:
        st.error(f"Error durante el inicio de sesión: {str(e)}")
        return None

def subir_archivo(file, user_id):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file.name}")
        blob.upload_from_string(file.getvalue(), content_type=file.type)
        return blob.public_url
    except Exception as e:
        st.error(f"Error al subir el archivo: {str(e)}")
        return None

def obtener_archivos_usuario(user_id):
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f"users/{user_id}/")
        return [blob.name.split('/')[-1] for blob in blobs if blob.name != f"users/{user_id}/"]
    except Exception as e:
        st.error(f"Error al obtener los archivos del usuario: {str(e)}")
        return []

def descargar_archivo(user_id, file_name):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file_name}")
        return blob.download_as_bytes()
    except Exception as e:
        st.error(f"Error al descargar el archivo: {str(e)}")
        return None

# Funciones de procesamiento de señales
def corregir_linea_base(datos):
    return datos - np.mean(datos)

def aplicar_filtro_pasabanda(datos, fs, fmin, fmax):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [fmin/nyq, fmax/nyq], btype='band')
    return signal.filtfilt(b, a, datos)

def calcular_espectro_fourier(datos, fs):
    n = len(datos)
    frecuencias = fftfreq(n, d=1/fs)[:n//2]
    amplitudes = np.abs(fft(datos))[:n//2]
    return frecuencias, amplitudes

def dividir_entre_gravedad(x, y, z):
    return x / 9.81, y / 9.81, z / 9.81

def analisis_hv_mejorado(x, y, z, fs, num_ventanas=20, tamano_ventana=2000, 
                        suavizado=True, fmin=0.1, fmax=10):
    try:
        rango_busqueda_pico = (0.1, 10)

        # 1. Preprocesamiento
        x = corregir_linea_base(x)
        y = corregir_linea_base(y)
        z = corregir_linea_base(z)

        # 2. Aplicar filtro pasa banda con parámetros ajustados
        x = aplicar_filtro_pasabanda(x, fs, fmin=fmin, fmax=fmax)
        y = aplicar_filtro_pasabanda(y, fs, fmin=fmin, fmax=fmax)
        z = aplicar_filtro_pasabanda(z, fs, fmin=fmin, fmax=fmax)

        # 3. Inicializar acumuladores
        n_fft = tamano_ventana // 2
        cociente_xz = np.zeros(n_fft)
        cociente_yz = np.zeros(n_fft)
        cociente_xz2 = np.zeros(n_fft)
        cociente_yz2 = np.zeros(n_fft)

        # 4. Procesamiento por ventanas
        for _ in range(num_ventanas):
            # Seleccionar ventana aleatoria con solapamiento
            max_start = len(x) - tamano_ventana
            nini = random.randint(int(0.1*max_start), int(0.9*max_start)) if max_start > 0 else 0
            x1 = x[nini:nini+tamano_ventana]
            y1 = y[nini:nini+tamano_ventana]
            z1 = z[nini:nini+tamano_ventana]

            # Calcular espectros de Fourier
            frecuencias, fx = calcular_espectro_fourier(x1, fs)
            _, fy = calcular_espectro_fourier(y1, fs)
            _, fz = calcular_espectro_fourier(z1, fs)

            # Suavizar espectros
            window_length = 11
            fx_suavizado = signal.savgol_filter(fx, window_length=window_length, polyorder=3)
            fy_suavizado = signal.savgol_filter(fy, window_length=window_length, polyorder=3)
            fz_suavizado = signal.savgol_filter(fz, window_length=window_length, polyorder=3)
            
            # Acumular cocientes
            with np.errstate(divide='ignore', invalid='ignore'):
                cociente_xz += fx_suavizado / fz_suavizado / num_ventanas
                cociente_yz += fy_suavizado / fz_suavizado / num_ventanas
                cociente_xz2 += (fx_suavizado / fz_suavizado)**2 / num_ventanas
                cociente_yz2 += (fy_suavizado / fz_suavizado)**2 / num_ventanas

        # 5. Calcular estadísticas
        var_xz = cociente_xz2 - cociente_xz**2
        std_xz = np.sqrt(np.abs(var_xz))

        var_yz = cociente_yz2 - cociente_yz**2
        std_yz = np.sqrt(np.abs(var_yz))

        # 6. Suavizado adaptativo
        if suavizado:
            window_length = 11
            hv_suavizado_xz = signal.savgol_filter(cociente_xz, window_length=window_length, polyorder=3)
            hv_suavizado_yz = signal.savgol_filter(cociente_yz, window_length=window_length, polyorder=3)
        else:
            hv_suavizado_xz = cociente_xz
            hv_suavizado_yz = cociente_yz

        # 7. Detección de pico fundamental en rango específico
        mask = (frecuencias >= rango_busqueda_pico[0]) & (frecuencias <= rango_busqueda_pico[1])
        
        try:
            indice_max_xz = np.nanargmax(hv_suavizado_xz[mask]) + np.argmax(mask)
            frecuencia_fundamental_xz = frecuencias[indice_max_xz]
        except:
            frecuencia_fundamental_xz = 0.0
            
        try:
            indice_max_yz = np.nanargmax(hv_suavizado_yz[mask]) + np.argmax(mask)
            frecuencia_fundamental_yz = frecuencias[indice_max_yz]
        except:
            frecuencia_fundamental_yz = 0.0

        # 8. Calcular parámetros de calidad
        calidad_xz = hv_suavizado_xz[indice_max_xz] / np.median(hv_suavizado_xz[mask])
        calidad_yz = hv_suavizado_yz[indice_max_yz] / np.median(hv_suavizado_yz[mask])

        return {
            'frecuencias': frecuencias,
            'hv_xz': cociente_xz,
            'hv_yz': cociente_yz,
            'hv_suavizado_xz': hv_suavizado_xz,
            'hv_suavizado_yz': hv_suavizado_yz,
            'std_xz': std_xz,
            'std_yz': std_yz,
            'frecuencia_fundamental_xz': frecuencia_fundamental_xz,
            'frecuencia_fundamental_yz': frecuencia_fundamental_yz,
            'periodo_fundamental_xz': 1/frecuencia_fundamental_xz if frecuencia_fundamental_xz > 0 else 0,
            'periodo_fundamental_yz': 1/frecuencia_fundamental_yz if frecuencia_fundamental_yz > 0 else 0,
            'calidad_xz': calidad_xz,
            'calidad_yz': calidad_yz,
            'mask_rango_valido': mask,
            'fx': fx_suavizado,
            'fy': fy_suavizado,
            'fz': fz_suavizado
        }
    except Exception as e:
        st.error(f"Error en el análisis H/V: {str(e)}")
        return None

# Funciones de visualización
def graficar_canales_individuales(x, y, z, fs, st):
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Canal X', 'Canal Y', 'Canal Z'))
    
    tiempo = np.arange(len(x)) / fs
    line_width = 1
    
    fig.add_trace(go.Scatter(x=tiempo, y=x, name='X', line=dict(width=line_width)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=y, name='Y', line=dict(width=line_width)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=z, name='Z', line=dict(width=line_width)), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, title_text="Canales filtrados")
    
    img_bytes = fig.to_image(format="png")
    st.download_button(label="Descargar gráfica como imagen", data=img_bytes, file_name="grafica_canales.png", mime="image/png")
    return fig

def graficar_hv(resultados_hv, st):
    if resultados_hv is None:
        st.error("No se pudieron obtener los resultados del análisis H/V.")
        return None

    fig = go.Figure()
    
    # Gráfico para H/V
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_xz'],
        mode='lines',
        name='H/V (X/Z)',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_yz'],
        mode='lines',
        name='H/V (Y/Z)',
        line=dict(color='red', width=2)
    ))
    
    # Líneas de desviación estándar
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_xz'] + resultados_hv['std_xz'],
        mode='lines',
        name='X/Z +σ',
        line=dict(color='lightblue', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_xz'] - resultados_hv['std_xz'],
        mode='lines',
        name='X/Z -σ',
        line=dict(color='lightblue', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_yz'] + resultados_hv['std_yz'],
        mode='lines',
        name='Y/Z +σ',
        line=dict(color='lightcoral', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_yz'] - resultados_hv['std_yz'],
        mode='lines',
        name='Y/Z -σ',
        line=dict(color='lightcoral', width=1, dash='dash')
    ))
    
    # Marcadores para las frecuencias fundamentales
    fig.add_trace(go.Scatter(
        x=[resultados_hv['frecuencia_fundamental_xz']],
        y=[resultados_hv['hv_suavizado_xz'][np.argmax(resultados_hv['hv_suavizado_xz'])]],
        mode='markers',
        name='f₀ (X/Z)',
        marker=dict(color='green', size=10, symbol='star')
    ))
    fig.add_trace(go.Scatter(
        x=[resultados_hv['frecuencia_fundamental_yz']],
        y=[resultados_hv['hv_suavizado_yz'][np.argmax(resultados_hv['hv_suavizado_yz'])]],
        mode='markers',
        name='f₀ (Y/Z)',
        marker=dict(color='orange', size=10, symbol='star')
    ))
    
    # Líneas de referencia
    fig.add_hline(y=2, line_dash="dot", line_color="gray", annotation_text="H/V = 2")
    fig.add_hline(y=4, line_dash="dot", line_color="gray", annotation_text="H/V = 4")
    
    # Configuración del gráfico
    fig.update_layout(
        title="Análisis H/V",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="H/V",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_range=[np.log10(0.1), np.log10(50)],  # Rango típico para H/V
        yaxis_range=[np.log10(0.1), np.log10(10)],  # Rango típico para H/V
        plot_bgcolor='white',
        width=800,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', dtick=0.30103)  # dtick para escala log
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    img_bytes = fig.to_image(format="png")
    st.download_button(label="Descargar gráfica como imagen", data=img_bytes, file_name="grafica_hv.png", mime="image/png")
    return fig

def graficar_espectro_fourier(resultados_hv, st):
    if resultados_hv is None:
        st.error("No se pudieron obtener los resultados del análisis H/V.")
        return None

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['fx'],
        mode='lines',
        name='Espectro X',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['fy'],
        mode='lines',
        name='Espectro Y',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['fz'],
        mode='lines',
        name='Espectro Z',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="Espectro de Fourier",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Amplitud",
        xaxis_type="log",
        yaxis_type="log",
        plot_bgcolor='white'
    )
    
    img_bytes = fig.to_image(format="png")
    st.download_button(label="Descargar gráfica como imagen", data=img_bytes, file_name="grafica_espectro_fourier.png", mime="image/png")
    return fig

def graficar_cocientes(resultados_hv, st):
    if resultados_hv is None:
        st.error("No se pudieron obtener los resultados del análisis H/V.")
        return None

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_xz'],
        mode='lines',
        name='Cociente X/Z',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_yz'],
        mode='lines',
        name='Cociente Y/Z',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Cocientes H/V",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Cociente H/V",
        xaxis_type="log",
        yaxis_type="log",
        plot_bgcolor='white'
    )
    
    img_bytes = fig.to_image(format="png")
    st.download_button(label="Descargar gráfica como imagen", data=img_bytes, file_name="grafica_cocientes_hv.png", mime="image/png")
    return fig

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
                user = iniciar_sesion(email, password)
                if user:
                    st.session_state.user = user
                    st.success("¡Sesión iniciada con éxito!")
                    st.rerun()

        with tab2:
            new_email = st.text_input("Nuevo correo electrónico")
            new_password = st.text_input("Nueva contraseña", type="password")
            if st.button("Registrarse"):
                user = registrar(new_email, new_password)
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
            file_url = subir_archivo(uploaded_file, st.session_state.user.uid)
            if file_url:
                st.success("¡Archivo subido exitosamente!")

        user_files = obtener_archivos_usuario(st.session_state.user.uid)
        selected_file = st.selectbox("Seleccione un archivo para analizar", user_files)

        if selected_file:
            file_content = descargar_archivo(st.session_state.user.uid, selected_file)
            if file_content:
                st.download_button(
                    label="Descargar archivo seleccionado",
                    data=file_content,
                    file_name=selected_file,
                    mime="text/csv"
                )
            
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
            
            suavizar_hv = st.sidebar.checkbox("Suavizar curva H/V", value=True)
            
            # Nuevos parámetros para el filtro pasa banda
            fmin = st.sidebar.number_input("Frecuencia mínima del filtro (Hz)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
            fmax = st.sidebar.number_input("Frecuencia máxima del filtro (Hz)", min_value=1.0, max_value=50.0, value=10.0, step=0.1)

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
                    suavizado=suavizar_hv,
                    fmin=fmin,
                    fmax=fmax
                )
                
                if resultados_hv is not None:
                    st.subheader(f"Canales filtrados ({fmin}-{fmax} Hz)")
                    fig_canales = graficar_canales_individuales(
                        datos_x, datos_y, datos_z, fs, st
                    )
                    st.plotly_chart(fig_canales)
                    
                    st.subheader("Espectro de Fourier")
                    fig_fourier = graficar_espectro_fourier(resultados_hv, st)
                    if fig_fourier is not None:
                        st.plotly_chart(fig_fourier)
                    
                    st.subheader("Cocientes H/V")
                    fig_cocientes = graficar_cocientes(resultados_hv, st)
                    if fig_cocientes is not None:
                        st.plotly_chart(fig_cocientes)
                    
                    st.subheader("Análisis H/V")
                    fig_hv = graficar_hv(resultados_hv, st)
                    if fig_hv is not None:
                        st.plotly_chart(fig_hv)
                    
                    st.subheader("Estadísticas del análisis H/V")
                    st.write(f"Frecuencia fundamental (X/Z): {resultados_hv.get('frecuencia_fundamental_xz', 'N/A')} Hz")
                    st.write(f"Frecuencia fundamental (Y/Z): {resultados_hv.get('frecuencia_fundamental_yz', 'N/A')} Hz")

                else:
                    st.error("No se pudo realizar el análisis H/V. Por favor, revise los datos de entrada y los parámetros.")

    st.sidebar.header("Instrucciones")
    st.sidebar.markdown("""
    1. Inicie sesión o cree una cuenta.
    2. Suba un archivo CSV o TXT para analizar.
    3. Seleccione un archivo de sus archivos subidos.
    4. Ajuste los parámetros de análisis en la barra lateral:
       - Frecuencia de muestreo
       - Número de ventanas para análisis H/V
       - Tamaño de ventana
       - Suavizado de la curva H/V
       - Frecuencias mínima y máxima del filtro pasa banda
    5. Haga clic en 'Analizar datos' para ver:
       - Canales filtrados individualmente
       - Espectro de Fourier
       - Cocientes H/V
       - Análisis H/V
       - Estadísticas del análisis
    6. Descargue los resultados si lo desea.
    """)

if __name__ == "__main__":
    main()

