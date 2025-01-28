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

def analisis_hv_mejorado(x, y, z, fs, num_ventanas=20, tamano_ventana=2000, suavizado=True, fmax=10):
    try:
        # 1. Corregir línea base y aplicar filtro pasa banda
        x = aplicar_filtro_pasabanda(corregir_linea_base(x), fs)
        y = aplicar_filtro_pasabanda(corregir_linea_base(y), fs)
        z = aplicar_filtro_pasabanda(corregir_linea_base(z), fs)
        
        # 2. Inicializar acumuladores para los cocientes
        cociente_xz = np.zeros(tamano_ventana // 2)
        cociente_yz = np.zeros(tamano_ventana // 2)
        cociente_xz2 = np.zeros(tamano_ventana // 2)
        cociente_yz2 = np.zeros(tamano_ventana // 2)
        
        # 3. Repetir el proceso para cada ventana
        for _ in range(num_ventanas):
            # 4. Seleccionar una ventana aleatoria
            nini = random.randint(0, len(x) - tamano_ventana)
            x1 = x[nini:nini+tamano_ventana]
            y1 = y[nini:nini+tamano_ventana]
            z1 = z[nini:nini+tamano_ventana]
            
            # 5. Calcular los espectros de Fourier
            frecuencias, fx = calcular_espectro_fourier(x1, fs)
            _, fy = calcular_espectro_fourier(y1, fs)
            _, fz = calcular_espectro_fourier(z1, fs)
            
            # 6. Limitar el análisis a frecuencias por debajo de fmax
            idx = frecuencias <= fmax
            frecuencias = frecuencias[idx]
            fx = fx[idx]
            fy = fy[idx]
            fz = fz[idx]
            
            # 7. Evitar división por cero
            fz[fz == 0] = np.finfo(float).eps  # Reemplazar ceros con un valor pequeño
            fx_div_fz = fx / fz
            fy_div_fz = fy / fz
            
            # 8. Acumular los cocientes y sus cuadrados
            cociente_xz += fx_div_fz / num_ventanas
            cociente_yz += fy_div_fz / num_ventanas
            cociente_xz2 += (fx_div_fz ** 2) / num_ventanas
            cociente_yz2 += (fy_div_fz ** 2) / num_ventanas
        
        # 9. Calcular la varianza y la desviación estándar
        var_xz = cociente_xz2 - cociente_xz**2
        std_xz = np.sqrt(var_xz)
        
        var_yz = cociente_yz2 - cociente_yz**2
        std_yz = np.sqrt(var_yz)
        
        # 10. Suavizar los resultados si es necesario
        if suavizado:
            hv_suavizado_xz = signal.savgol_filter(cociente_xz, window_length=11, polyorder=3)
            hv_suavizado_yz = signal.savgol_filter(cociente_yz, window_length=11, polyorder=3)
        else:
            hv_suavizado_xz = cociente_xz
            hv_suavizado_yz = cociente_yz
        
        # 11. Encontrar la frecuencia fundamental dentro del rango limitado
        indice_max_xz = np.argmax(hv_suavizado_xz)
        frecuencia_fundamental_xz = frecuencias[indice_max_xz]
        periodo_fundamental_xz = 1 / frecuencia_fundamental_xz
        
        indice_max_yz = np.argmax(hv_suavizado_yz)
        frecuencia_fundamental_yz = frecuencias[indice_max_yz]
        periodo_fundamental_yz = 1 / frecuencia_fundamental_yz
        
        # 12. Retornar los resultados
        return {
            'frecuencias': frecuencias,
            'hv_xz': cociente_xz,
            'hv_yz': cociente_yz,
            'hv_suavizado_xz': hv_suavizado_xz,
            'hv_suavizado_yz': hv_suavizado_yz,
            'hv_mas_std_xz': cociente_xz + std_xz,
            'hv_menos_std_xz': cociente_xz - std_xz,
            'hv_mas_std_yz': cociente_yz + std_yz,
            'hv_menos_std_yz': cociente_yz - std_yz,
            'frecuencia_fundamental_xz': frecuencia_fundamental_xz,
            'frecuencia_fundamental_yz': frecuencia_fundamental_yz,
            'periodo_fundamental_xz': periodo_fundamental_xz,
            'periodo_fundamental_yz': periodo_fundamental_yz
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
    
    # Gráfico para x/z
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_suavizado_xz'],
        mode='lines',
        name='H/V (X/Z)',
        line=dict(color='blue', width=2)
    ))
    
    # Gráfico para y/z
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
        y=resultados_hv['hv_mas_std_xz'],
        mode='lines',
        name='m+s (X/Z)',
        line=dict(color='gray', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_menos_std_xz'],
        mode='lines',
        name='m-s (X/Z)',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_mas_std_yz'],
        mode='lines',
        name='m+s (Y/Z)',
        line=dict(color='lightgray', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=resultados_hv['frecuencias'],
        y=resultados_hv['hv_menos_std_yz'],
        mode='lines',
        name='m-s (Y/Z)',
        line=dict(color='lightgray', width=1, dash='dash')
    ))
    
    # Marcadores para las frecuencias fundamentales
    fig.add_trace(go.Scatter(
        x=[resultados_hv['frecuencia_fundamental_xz']],
        y=[resultados_hv['hv_suavizado_xz'][np.argmax(resultados_hv['hv_suavizado_xz'])]],
        mode='markers',
        name='Frecuencia fundamental (X/Z)',
        marker=dict(color='green', size=10, symbol='star')
    ))
    fig.add_trace(go.Scatter(
        x=[resultados_hv['frecuencia_fundamental_yz']],
        y=[resultados_hv['hv_suavizado_yz'][np.argmax(resultados_hv['hv_suavizado_yz'])]],
        mode='markers',
        name='Frecuencia fundamental (Y/Z)',
        marker=dict(color='orange', size=10, symbol='star')
    ))
    
    # Configuración del gráfico
    fig.update_layout(
        title="Análisis H/V",
        xaxis_title="f, Hz",
        yaxis_title="H/V",
        xaxis_type="log",
        yaxis_type="log",
        plot_bgcolor='white',
        width=800,
        height=500
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    img_bytes = fig.to_image(format="png")
    st.download_button(label="Descargar gráfica como imagen", data=img_bytes, file_name="grafica_hv.png", mime="image/png")
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

