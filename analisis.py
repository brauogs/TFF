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
from scipy.ndimage import gaussian_filter1d

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

# Una vez
def aplicar_filtro_pasabanda(datos, fs, fmin=0.05, fmax=10):
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

def preprocesar_movil(datos, fs):
    """Preprocesamiento adicional para datos de dispositivos móviles"""
    try:
        # 1. Filtro notch para eliminar interferencia eléctrica (50/60 Hz)
        notch_freq = 60 if fs > 100 else 50
        b, a = signal.iirnotch(notch_freq, 30, fs)
        datos_filtrados = signal.filtfilt(b, a, datos)
        
        # 2. Filtro pasa-altos adicional para eliminar deriva
        b_hp, a_hp = signal.butter(2, 0.5/(fs/2), btype='high')
        return signal.filtfilt(b_hp, a_hp, datos_filtrados)
    except Exception as e:
        st.error(f"Error en preprocesamiento móvil: {str(e)}")
        return datos

def analisis_hv_mejorado(x, y, z, fs, num_ventanas=20, tamano_ventana=2000, device_type='accelerometer'):
    try:
        # Ajustar parámetros según tipo de dispositivo
        if device_type == 'mobile':
            fmin, fmax = 0.1, 20  # Rango optimizado para móviles
            rango_busqueda_pico = (0.5, 5)  # Rango de búsqueda de frecuencia fundamental
        else:
            fmin, fmax = 0.05, 10  # Rango tradicional para acelerómetros
            rango_busqueda_pico = (0.1, 10)

        # 1. Preprocesamiento específico para móviles
        if device_type == 'mobile':
            x = preprocesar_movil(corregir_linea_base(x), fs)
            y = preprocesar_movil(corregir_linea_base(y), fs)
            z = preprocesar_movil(corregir_linea_base(z), fs)
        else:
            x = corregir_linea_base(x)
            y = corregir_linea_base(y)
            z = corregir_linea_base(z)

        # 2. Aplicar filtro pasa banda con parámetros ajustados
        x = aplicar_filtro_pasabanda(x, fs, fmin=fmin, fmax=fmax)
        y = aplicar_filtro_pasabanda(y, fs, fmin=fmin, fmax=fmax)
        z = aplicar_filtro_pasabanda(z, fs, fmin=fmin, fmax=fmax)

        # Windows for H/V
        # este tamaño de ventana de 1000 puntos o 10 segundos se ve que funciona mejor y esta dentro
        # de lo recomendado en la literatura
        wl = int(1000)
        # aumente el numero de ventanas a 100, se ve que como el sensor de iphone tiene menos sensibilidad hay factores
        # influyen mas que con el acelerometro
        # lo que habría que hacer es muestrear a otras horas y otros días
        # eso lo voy a hacer ya en el futuro
        # para tu informe simplemente menciona que con 100 ventanas se obtienen resultados estables
        nw = 100

        # Inicializar acumuladores
        hxz = np.zeros(wl // 2 + 1)
        hyz = np.zeros(wl // 2 + 1)
        hxz2 = np.zeros(wl // 2 + 1)
        hyz2 = np.zeros(wl // 2 + 1)

        # Procesamiento por ventanas - 
        for k in range(1, nw):
            random_number = random.uniform(1, len(x) - wl - 1)
            # Indices
            i = int(random_number)
            j = int(i + wl)
            # Extract values between indices i and j
            ax = x[i:j]
            ay = y[i:j]
            az = z[i:j]

            # La amplitud se calcula con la transformada de Fourier. 
            # Compute the amplitude spectrum
            # también hay algo raro con la escala de la fft, normalmente luego hay unas constantes involucradas que a veces
            # las rutinas no las hacen transparantes, sin embargo esto no afecta en los cocientes H/V ya que el factor
            # va aparecer en el nunerador y en el denominador
            ax_amplitude = np.abs(fft(ax))[:wl // 2 + 1]
            ay_amplitude = np.abs(fft(ay))[:wl // 2 + 1]
            az_amplitude = np.abs(fft(az))[:wl // 2 + 1]

            # Smooth the amplitude spectrum
            # parte del problema era el suavizado que estabas usando
            # esl filtro gaussiano parece mas razonable hay que dejar sigma=0.4
            # en el futuro voy a poner la funcion de suavizado que nosotros usamos
            ax_amplitude = gaussian_filter1d(ax_amplitude, sigma=0.4)
            ay_amplitude = gaussian_filter1d(ay_amplitude, sigma=0.4)
            az_amplitude = gaussian_filter1d(az_amplitude, sigma=0.4)

            # Compute the frequency bins
            freq = np.fft.fftfreq(wl, d=1/fs)[:wl // 2 + 1]

            # H/V computations
            hxz += ax_amplitude / az_amplitude / nw
            hyz += ay_amplitude / az_amplitude / nw
            hxz2 += (ax_amplitude ** 2) / (az_amplitude ** 2) / nw
            hyz2 += (ay_amplitude ** 2) / (az_amplitude ** 2) / nw


        # 5. Calcular estadísticas
        var_xz = hxz2 - hxz**2
        std_xz = np.sqrt(np.abs(var_xz))  # Valor absoluto para evitar valores negativos por errores numéricos

        var_yz = hyz2 - hyz**2
        std_yz = np.sqrt(np.abs(var_yz))

        # 7. Detección de pico fundamental en rango específico
        mask = (freq >= rango_busqueda_pico[0]) & (freq <= rango_busqueda_pico[1])
        
        try:
            indice_max_xz = np.argmax(hxz[mask]) + np.argmax(mask)
            frecuencia_fundamental_xz = freq[indice_max_xz]
        except:
            frecuencia_fundamental_xz = 0.0
            
        try:
            indice_max_yz = np.argmax(hyz[mask]) + np.argmax(mask)
            frecuencia_fundamental_yz = freq[indice_max_yz]
        except:
            frecuencia_fundamental_yz = 0.0

        # 8. Calcular parámetros de calidad
        calidad_xz = np.max(hxz) / np.median(hxz)
        calidad_yz = np.max(hyz) / np.median(hyz)

        return {
            'frecuencias': freq,
            'hv_xz': hxz,
            'hv_yz': hyz,
            'hv_suavizado_xz': hxz,
            'hv_suavizado_yz': hyz,
            'std_xz': np.sqrt(np.abs(hxz2 - hxz**2)),
            'std_yz': np.sqrt(np.abs(hyz2 - hyz**2)),
            'frecuencia_fundamental_xz': freq[np.argmax(hxz)],
            'frecuencia_fundamental_yz': freq[np.argmax(hyz)],
            'periodo_fundamental_xz': 1/freq[np.argmax(hxz)] if freq[np.argmax(hxz)] > 0 else 0,
            'periodo_fundamental_yz': 1/freq[np.argmax(hyz)] if freq[np.argmax(hyz)] > 0 else 0,
            'calidad_xz': calidad_xz,
            'calidad_yz': calidad_yz,
            'mask_rango_valido': (freq >= 0.1) & (freq <= 10),
            'fx': ax_amplitude,
            'fy': ay_amplitude,
            'fz': az_amplitude
        }
    except Exception as e:
        st.error(f"Error en el análisis H/V: {str(e)}")
        return None

# Funciones de visualización
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
                
                
                # Calcular con los espectros suavizados.
                
                resultados_hv = analisis_hv_mejorado(
                    datos_x, datos_y, datos_z,
                    fs=fs,
                    num_ventanas=num_ventanas,
                    tamano_ventana=tamano_ventana,
                    device_type=device_type
                )
                
                if resultados_hv is not None:
                    st.subheader("Canales filtrados (0.05-1.5 Hz)")
                    fig_canales = graficar_canales_individuales(
                        datos_x, datos_y, datos_z, fs, st, device_type
                    )
                    st.plotly_chart(fig_canales)
                    
                    st.subheader("Análisis H/V")
                    fig_hv = graficar_hv(resultados_hv, st)
                    if fig_hv is not None:
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
       - Tipo de dispositivo
       - Suavizado de la curva H/V
    5. Haga clic en 'Analizar datos' para ver:
       - Canales filtrados individualmente
       - Análisis H/V
       - Estadísticas del análisis
    6. Descargue los resultados si lo desea.
    """)

if __name__ == "__main__":
    main()

