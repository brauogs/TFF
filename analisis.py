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
from io import BytesIO

st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

# Funciones de Firebase (sin cambios)
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

def download_file(user_id, file_name):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{user_id}/{file_name}")
        content = blob.download_as_bytes()
        return content
    except Exception as e:
        st.error(f"Error al descargar el archivo: {str(e)}")
        return None

# Funciones de procesamiento de señales
def corregir_linea_base(datos):
    """Corrige la línea base de los datos"""
    return datos - np.mean(datos)

def aplicar_filtro_pasabanda(datos, fs, fmin=0.05, fmax=10):
    """Aplica un filtro pasa banda entre fmin y fmax Hz"""
    nyq = 0.5 * fs
    b, a = signal.butter(4, [fmin/nyq, fmax/nyq], btype='band')
    return signal.filtfilt(b, a, datos)

def calcular_espectro_fourier(datos):
    """Calcula el espectro de Fourier de los datos"""
    return np.abs(fft(datos))

def dividir_entre_gravedad(x, y, z):
    return x / 9.81, y / 9.81, z / 9.81

# Función analisis_hv modificada
def analisis_hv(x, y, z, fs, num_ventanas=20, tamano_ventana=2000, fmin=0.1, fmax=20):
    """
    Realiza el análisis H/V siguiendo el método especificado:
    1. Corrección de línea base y filtrado
    2. Selección de ventanas aleatorias
    3. Cálculo de espectros de Fourier
    4. Cálculo de cocientes H/V
    5. Análisis estadístico
    """
    # Corrección de línea base y filtrado
    x = aplicar_filtro_pasabanda(corregir_linea_base(x), fs, fmin, fmax)
    y = aplicar_filtro_pasabanda(corregir_linea_base(y), fs, fmin, fmax)
    z = aplicar_filtro_pasabanda(corregir_linea_base(z), fs, fmin, fmax)
    
    # Inicialización de acumuladores
    Cociente_xz = np.zeros(tamano_ventana // 2)
    Cociente_yz = np.zeros(tamano_ventana // 2)
    Cociente_xz2 = np.zeros(tamano_ventana // 2)
    Cociente_yz2 = np.zeros(tamano_ventana // 2)
    
    frecuencias = np.fft.fftfreq(tamano_ventana, d=1/fs)[:tamano_ventana//2]
    frecuencias_validas = (frecuencias >= fmin) & (frecuencias <= fmax)
    
    # Para cada ventana
    for _ in range(num_ventanas):
        # Selección aleatoria de ventana
        nini = random.randint(0, len(x) - tamano_ventana)
        x1 = x[nini:nini+tamano_ventana]
        y1 = y[nini:nini+tamano_ventana]
        z1 = z[nini:nini+tamano_ventana]
        
        # Cálculo de espectros de Fourier
        fx = calcular_espectro_fourier(x1)[:tamano_ventana//2]
        fy = calcular_espectro_fourier(y1)[:tamano_ventana//2]
        fz = calcular_espectro_fourier(z1)[:tamano_ventana//2]
        
        # Acumular los valores solo para frecuencias válidas
        Cociente_xz[frecuencias_validas] += (fx[frecuencias_validas] / fz[frecuencias_validas]) / num_ventanas
        Cociente_yz[frecuencias_validas] += (fy[frecuencias_validas] / fz[frecuencias_validas]) / num_ventanas
        Cociente_xz2[frecuencias_validas] += (fx[frecuencias_validas] / fz[frecuencias_validas]) ** 2 / num_ventanas
        Cociente_yz2[frecuencias_validas] += (fy[frecuencias_validas] / fz[frecuencias_validas]) ** 2 / num_ventanas

    # Cálculo de varianza y desviación estándar
    Var_xz = Cociente_xz2 - (Cociente_xz ** 2)
    Var_yz = Cociente_yz2 - (Cociente_yz ** 2)
    Desviacion_xz = np.sqrt(np.abs(Var_xz))  # Usar valor absoluto para evitar warnings
    Desviacion_yz = np.sqrt(np.abs(Var_yz))
    
    # Cálculo del ratio H/V promedio
    hv = np.sqrt((Cociente_xz ** 2 + Cociente_yz ** 2) / 2)
    hv_std = np.sqrt((Desviacion_xz ** 2 + Desviacion_yz ** 2) / 2)
    
    # Suavizado de la curva H/V
    hv_suavizado = signal.savgol_filter(hv, window_length=11, polyorder=3)
    
    # Cálculo de la frecuencia fundamental
    indice_max = np.argmax(hv_suavizado[frecuencias_validas])
    frecuencia_fundamental = frecuencias[frecuencias_validas][indice_max]

    # Cálculo del periodo fundamental
    periodo_fundamental = 1 / frecuencia_fundamental

    # Estadísticas globales
    estadisticas_globales = {
        'promedio_xz': np.mean(Cociente_xz[frecuencias_validas]),
        'promedio_yz': np.mean(Cociente_yz[frecuencias_validas]),
        'std_xz': np.mean(Desviacion_xz[frecuencias_validas]),
        'std_yz': np.mean(Desviacion_yz[frecuencias_validas])
    }
    
    return {
        'frecuencias': frecuencias[frecuencias_validas],
        'hv': hv[frecuencias_validas],
        'hv_suavizado': hv_suavizado[frecuencias_validas],
        'hv_mas_std': (hv + hv_std)[frecuencias_validas],
        'hv_menos_std': (hv - hv_std)[frecuencias_validas],
        'media_xz': Cociente_xz[frecuencias_validas],
        'media_yz': Cociente_yz[frecuencias_validas],
        'std_xz': Desviacion_xz[frecuencias_validas],
        'std_yz': Desviacion_yz[frecuencias_validas],
        'cocientes_xz': [Cociente_xz[frecuencias_validas], Cociente_xz2[frecuencias_validas]],
        'cocientes_yz': [Cociente_yz[frecuencias_validas], Cociente_yz2[frecuencias_validas]],
        'frecuencia_fundamental': frecuencia_fundamental,
        'periodo_fundamental': periodo_fundamental,
        'estadisticas_globales': estadisticas_globales
    }

# Funciones de visualización (con cambios)
def graficar_canales_individuales(x, y, z, fs, st):
    """Grafica cada canal de forma individual después del filtrado"""
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=('Canal X', 'Canal Y', 'Canal Z'))
    
    tiempo = np.arange(len(x)) / fs
    
    fig.add_trace(go.Scatter(x=tiempo, y=x, name='X'), row=1, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=y, name='Y'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=z, name='Z'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, title_text="Canales filtrados (0.05-10 Hz)")
    
    # Add download button for graph image
    img_bytes = fig.to_image(format="png")
    btn = st.download_button(
        label="Descargar gráfica como imagen",
        data=img_bytes,
        file_name="grafica_canales.png",
        mime="image/png"
    )
    return fig

def graficar_ventana(x1, y1, z1, fs, nini, st):
    """Grafica una ventana seleccionada de los datos"""
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=(
                            f'Canal X (Ventana: {nini}-{nini+len(x1)})', 
                            f'Canal Y (Ventana: {nini}-{nini+len(y1)})', 
                            f'Canal Z (Ventana: {nini}-{nini+len(z1)})'
                        ))
    
    tiempo = np.arange(len(x1)) / fs
    frecuencias = np.fft.fftfreq(len(x1), d=1/fs)[:len(x1)//2]
    
    # Señales en tiempo
    fig.add_trace(go.Scatter(x=tiempo, y=x1, name='X'), row=1, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=y1, name='Y'), row=2, col=1)
    fig.add_trace(go.Scatter(x=tiempo, y=z1, name='Z'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text=f"Ventana seleccionada ({frecuencias[0]:.2f} Hz - {frecuencias[-1]:.2f} Hz)")
    
    # Add download button for graph image
    img_bytes = fig.to_image(format="png")
    btn = st.download_button(
        label="Descargar gráfica como imagen",
        data=img_bytes,
        file_name="grafica_ventana.png",
        mime="image/png"
    )
    return fig

def graficar_hv(resultados_hv, st):
    """Genera el gráfico H/V similar al mostrado en la imagen"""
    fig = go.Figure()
    
    # Línea media
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv'],
        mode='lines',
        name='H/V',
        line=dict(color='blue', width=1)
    ))
    
    # Línea suavizada
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado'],
        mode='lines',
        name='H/V suavizado',
        line=dict(color='red', width=2)
    ))
    
    # Líneas de desviación estándar
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_mas_std'],
        mode='lines',
        name='m+s',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_menos_std'],
        mode='lines',
        name='m-s',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    # Marcar la frecuencia fundamental
    if resultados_hv['frecuencia_fundamental']:
        fig.add_trace(go.Scatter(
            x=[resultados_hv['frecuencia_fundamental']],
            y=[resultados_hv['hv_suavizado'][np.argmax(resultados_hv['hv_suavizado'])]],
            mode='markers',
            name='Frecuencia fundamental',
            marker=dict(color='green', size=10, symbol='star')
        ))
    
    # Configuración del layout
    fig.update_layout(
        title="Análisis H/V",
        xaxis_title="f, Hz",
        yaxis_title="H/V",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_range=[-1.3, 1],  # Ajustado para mostrar mejor el rango de 0.05 a 10 Hz
        yaxis_range=[-1, 1],
        plot_bgcolor='white',
        width=800,
        height=500
    )
    
    # Agregar cuadrícula
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Add download button for graph image
    img_bytes = fig.to_image(format="png")
    btn = st.download_button(
        label="Descargar gráfica como imagen",
        data=img_bytes,
        file_name="grafica_hv.png",
        mime="image/png"
    )
    return fig

# Función principal
def main():
    st.title("Análisis del Acelerograma")
    st.sidebar.image("logoUAMSis.png", use_container_width=True)

    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Iniciar sesión", "Registrarse"])
        
        with tab1:
            email = st.text_input("Correo electrónicos")
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

            # Add download button for the selected file
            file_content = download_file(st.session_state.user.uid, selected_file)
            if file_content:
                st.download_button(
                    label="Descargar archivo seleccionado",
                    data=file_content,
                    file_name=selected_file,
                    mime="text/csv"
                )

            # Parámetros de análisis
            st.sidebar.header("Parámetros de análisis")
            fs = st.sidebar.number_input("Frecuencia de muestreo (Hz)", min_value=1, value=100)
            num_ventanas = st.sidebar.number_input("Número de ventanas para análisis H/V", min_value=1, max_value=100, value=20)
            tamano_ventana = st.sidebar.number_input("Tamaño de ventana (puntos)", min_value=100, max_value=10000, value=2000)
            fmin = st.sidebar.number_input("Frecuencia mínima de análisis (Hz)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
            fmax = st.sidebar.number_input("Frecuencia máxima de análisis (Hz)", min_value=1.0, max_value=50.0, value=20.0, step=0.1)
            dividir_por_g = st.sidebar.checkbox("Dividir datos por 9.81 (gravedad)", value=False)

            if st.button("Analizar datos"):
                # Obtener y procesar datos
                datos_x = df['x'].values
                datos_y = df['y'].values
                datos_z = df['z'].values

                # Si el checkbox está marcado, dividir entre 9.81
                if dividir_por_g:
                    datos_x = datos_x / 9.81
                    datos_y = datos_y / 9.81
                    datos_z = datos_z / 9.81
                    
                    # Crear DataFrame para mostrar comparación de datos
                    df_dividido = df.copy()
                    df_dividido['x'] = df['x'] / 9.81
                    df_dividido['y'] = df['y'] / 9.81
                    df_dividido['z'] = df['z'] / 9.81
                    
                    # Mostrar las primeras 10 filas comparando los datos originales y los divididos
                    st.subheader("Comparación de datos originales y divididos entre 9.81")
                    df_comparacion = df.head(10).copy()
                    df_comparacion['x_dividido'] = df_dividido['x'].head(10)
                    df_comparacion['y_dividido'] = df_dividido['y'].head(10)
                    df_comparacion['z_dividido'] = df_dividido['z'].head(10)
                    st.write(df_comparacion)
                
                # Corrección de línea base y filtrado
                x_proc = aplicar_filtro_pasabanda(corregir_linea_base(datos_x), fs)
                y_proc = aplicar_filtro_pasabanda(corregir_linea_base(datos_y), fs)
                z_proc = aplicar_filtro_pasabanda(corregir_linea_base(datos_z), fs)
                
                # Mostrar canales filtrados
                st.subheader("Canales filtrados (0.05-10 Hz)")
                fig_canales = graficar_canales_individuales(x_proc, y_proc, z_proc, fs, st)
                st.plotly_chart(fig_canales)
                
                # Realizar análisis H/V
                resultados_hv = analisis_hv(
                    x_proc, y_proc, z_proc,
                    fs=fs,
                    num_ventanas=num_ventanas,
                    tamano_ventana=tamano_ventana,
                    fmin=fmin,
                    fmax=fmax
                )
                
                # Mostrar ejemplo de ventana seleccionada
                st.subheader("Ejemplo de ventana seleccionada")
                nini = random.randint(0, len(x_proc) - tamano_ventana)
                x1 = x_proc[nini:nini+tamano_ventana]
                y1 = y_proc[nini:nini+tamano_ventana]
                z1 = z_proc[nini:nini+tamano_ventana]
                
                fig_ventana = graficar_ventana(x1, y1, z1, fs, nini, st)
                st.plotly_chart(fig_ventana)
                
                # Mostrar gráfico H/V
                st.subheader("Análisis H/V")
                fig_hv = graficar_hv(resultados_hv, st)
                st.plotly_chart(fig_hv)
                
                # Mostrar estadísticas
                st.subheader("Estadísticas del análisis H/V")
                st.write(f"Frecuencia fundamental: {resultados_hv['frecuencia_fundamental']:.2f} Hz")
                st.write(f"Periodo fundamental: {resultados_hv['periodo_fundamental']:.2f} s")
                if resultados_hv['frecuencia_fundamental'] < fmin or resultados_hv['frecuencia_fundamental'] > fmax:
                    st.warning(f"La frecuencia fundamental está fuera del rango de análisis especificado ({fmin}-{fmax} Hz).")

                st.write("Estadísticas globales de los cocientes de amplitud:")
                st.write(f"Promedio x/z: {resultados_hv['estadisticas_globales']['promedio_xz']:.4f}")
                st.write(f"Desviación estándar x/z: {resultados_hv['estadisticas_globales']['std_xz']:.4f}")
                st.write(f"Promedio y/z: {resultados_hv['estadisticas_globales']['promedio_yz']:.4f}")
                st.write(f"Desviación estándar y/z: {resultados_hv['estadisticas_globales']['std_yz']:.4f}")

                st.write("\nEstadísticas detalladas:")
                st.write(f"Promedio x/z: {np.mean(resultados_hv['media_xz']):.4f}")
                st.write(f"Promedio y/z: {np.mean(resultados_hv['media_yz']):.4f}")
                st.write(f"Promedio + Desviación estándar x/z: {np.mean(resultados_hv['media_xz'] + resultados_hv['std_xz']):.4f}")
                st.write(f"Promedio - Desviación estándar x/z: {np.mean(resultados_hv['media_xz'] - resultados_hv['std_xz']):.4f}")
                st.write(f"Promedio + Desviación estándar y/z: {np.mean(resultados_hv['media_yz'] + resultados_hv['std_yz']):.4f}")
                st.write(f"Promedio - Desviación estándar y/z: {np.mean(resultados_hv['media_yz'] - resultados_hv['std_yz']):.4f}")

                # Mostrar cocientes adicionales
                st.subheader("Cocientes adicionales")
                for i in range(1, len(resultados_hv['cocientes_xz'])):
                    st.write(f"Cociente {i+1} x/z: {np.mean(resultados_hv['cocientes_xz'][i]):.4f}")
                    st.write(f"Cociente {i+1} y/z: {np.mean(resultados_hv['cocientes_yz'][i]):.4f}")

    st.sidebar.header("Instrucciones")
    st.sidebar.markdown("""
    1. Inicie sesión o cree una cuenta.
    2. Suba un archivo CSV o TXT para analizar.
    3. Seleccione un archivo de sus archivos subidos.
    4. Ajuste los parámetros de análisis en la barra lateral:
       - Frecuencia de muestreo
       - Número de ventanas para análisis H/V
       - Tamaño de ventana
       - Frecuencia mínima de análisis
       - Frecuencia máxima de análisis
    5. Haga clic en 'Analizar datos' para ver:
       - Canales filtrados individualmente
       - Ventana seleccionada aleatoriamente
       - Análisis H/V
       - Estadísticas completas
    """)

if __name__ == "__main__":
    main()

