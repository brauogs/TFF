import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import pyperclip
import random

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
                if diferencia_tiempo == 0:
                    fs = fs_predeterminada
                else:
                    fs = 1 / diferencia_tiempo
        except Exception as e:
            st.warning(f"Error al procesar la columna de tiempo: {str(e)}. Se usará un índice numérico en su lugar.")
            df[columna_tiempo] = pd.to_numeric(df.index)
            fs = fs_predeterminada
    else:
        # Si no hay columna de tiempo, usamos el índice como tiempo
        st.warning("No se encontró una columna de tiempo válida. Se usará el índice como tiempo.")
        df['tiempo'] = pd.to_numeric(df.index)
        columna_tiempo = 'tiempo'
        fs = fs_predeterminada

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
    inicio_maximo = len(datos) - longitud_seccion
    indices_inicio = sorted(random.sample(range(inicio_maximo), num_secciones))
    return [(inicio, inicio + longitud_seccion) for inicio in indices_inicio]

def graficar_ratio_y_promedio(resultados, fs, tipo_ratio):
    numerador = 'x' if tipo_ratio == 'X/Z' else 'y'
    denominador = 'z'
    
    ratio = resultados[numerador]['magnitudes_fft'] / resultados[denominador]['magnitudes_fft']
    frecuencias = resultados[numerador]['frecuencias_fft']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frecuencias, y=ratio, name=f'Ratio {tipo_ratio}'))
    
    # Calcular y graficar el promedio móvil
    tamano_ventana = 10
    promedio_movil = np.convolve(ratio, np.ones(tamano_ventana)/tamano_ventana, mode='valid')
    fig.add_trace(go.Scatter(x=frecuencias[tamano_ventana-1:], y=promedio_movil, name=f'Promedio Móvil {tipo_ratio}'))
    
    fig.update_layout(title=f'Ratio {tipo_ratio} y Promedio Móvil',
                      xaxis_title='Frecuencia (Hz)',
                      yaxis_title='Ratio',
                      height=600, width=1000)
    return fig

st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

st.title("Análisis del Acelerograma")

archivo_subido = st.file_uploader("Elige un archivo", type="csv")

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.write("Visualizar datos:")
    st.write(df.head())

    # Verificar las columnas existentes
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

        if canal_seleccionado != 'Todos los canales':
            if st.button(f'Copiar el espectro de Fourier del canal {canal_seleccionado.upper()} al portapapeles'):
                copiar_espectro_al_portapapeles(resultados, canal_seleccionado)

        st.subheader(f"Rutinas FFT en {num_rutinas_fft} secciones aleatorias")
        for canal in canales:
            secciones = seleccionar_secciones_aleatorias(resultados[canal]['serie_filtrada'], fs, num_secciones=num_rutinas_fft)
            
            # Graficar secciones seleccionadas
            fig_secciones = go.Figure()
            fig_secciones.add_trace(go.Scatter(x=df[columna_tiempo], 
                                               y=resultados[canal]['serie_filtrada'], 
                                               name=f"Señal filtrada {canal.upper()}"))
            for i, (inicio, fin) in enumerate(secciones):
                fig_secciones.add_trace(go.Scatter(x=df[columna_tiempo].iloc[inicio:fin], 
                                                   y=resultados[canal]['serie_filtrada'][inicio:fin], 
                                                   name=f"Sección {i+1}",
                                                   line=dict(width=3)))
            fig_secciones.update_layout(height=400, width=1000, title_text=f"Secciones seleccionadas para FFT - Canal {canal.upper()}")
            st.plotly_chart(fig_secciones)

            # Calcular y graficar FFT para cada sección
            fig_fft = make_subplots(rows=num_rutinas_fft, cols=1, 
                                    subplot_titles=[f"FFT Sección {i+1}" for i in range(num_rutinas_fft)])
            for i, (inicio, fin) in enumerate(secciones):
                datos_seccion = resultados[canal]['serie_filtrada'][inicio:fin]
                frecuencias, magnitudes = calcular_fft(datos_seccion, fs)
                fig_fft.add_trace(go.Scatter(x=frecuencias, y=magnitudes, name=f"FFT Sección {i+1}"), row=i+1, col=1)
                fig_fft.update_xaxes(title_text="Frecuencia (Hz)", row=i+1, col=1)
                fig_fft.update_yaxes(title_text="Magnitud", row=i+1, col=1)
            fig_fft.update_layout(height=300*num_rutinas_fft, width=1000, title_text=f"FFT de Secciones Aleatorias - Canal {canal.upper()}")
            st.plotly_chart(fig_fft)

        if canal_seleccionado == 'Todos los canales':
            st.subheader("Ratios X/Z e Y/Z")
            fig_ratio_xz = graficar_ratio_y_promedio(resultados, fs, 'X/Z')
            st.plotly_chart(fig_ratio_xz)
            fig_ratio_yz = graficar_ratio_y_promedio(resultados, fs, 'Y/Z')
            st.plotly_chart(fig_ratio_yz)

        # Preparar datos para descargar
        salida = io.StringIO()
        datos_para_csv = {columna_tiempo: df[columna_tiempo]}
        longitud_maxima = len(datos_para_csv[columna_tiempo])

        for canal in canales:
            # Asegurar que todos los arrays tengan la misma longitud rellenando con valores NaN
            serie_tiempo = np.pad(resultados[canal]['serie_tiempo'], (0, longitud_maxima - len(resultados[canal]['serie_tiempo'])), mode='constant', constant_values=np.nan)
            serie_filtrada = np.pad(resultados[canal]['serie_filtrada'], (0, longitud_maxima - len(resultados[canal]['serie_filtrada'])), mode='constant', constant_values=np.nan)
            frecuencias_fft = np.pad(resultados[canal]['frecuencias_fft'], (0, longitud_maxima - len(resultados[canal]['frecuencias_fft'])), mode='constant', constant_values=np.nan)
            magnitudes_fft = np.pad(resultados[canal]['magnitudes_fft'], (0, longitud_maxima - len(resultados[canal]['magnitudes_fft'])), mode='constant', constant_values=np.nan)
            
            datos_para_csv.update({
                f'{canal}_original': serie_tiempo,
                f'{canal}_filtrado': serie_filtrada,
                f'{canal}_frecuencia_fft': frecuencias_fft,
                f'{canal}_magnitud_fft': magnitudes_fft
            })

        # Crear DataFrame y guardar como CSV
        df_para_csv = pd.DataFrame(datos_para_csv)
        df_para_csv.to_csv(salida, index=False)

        st.download_button(
            label="Descargar datos procesados",
            data=salida.getvalue(),
            file_name="datos_procesados.csv",
            mime="text/csv"
        )

else:
    st.write("Sube un archivo CSV.")

st.sidebar.header("Instrucciones")
st.sidebar.markdown("""
1. Sube el archivo en formato CSV. El archivo debe contener columnas para los canales 'x', 'y', y 'z'.
   Si existe, también puede incluir una columna de tiempo ('marca_tiempo', 'timestamp', o 'tiempo').
2. Selecciona el canal a analizar (x, y, z, o todos los canales).
3. Ajusta los parámetros de filtrado y de taper.
4. Especifica el número de rutinas FFT a realizar.
5. Haz clic en 'Analizar datos'.
6. Observa las secciones seleccionadas y sus respectivos FFT.
7. Si seleccionaste 'Todos los canales', observa los ratios X/Z e Y/Z y sus promedios.
8. Usa el botón para copiar el espectro de Fourier al portapapeles (solo para canales individuales).
9. Descarga los datos procesados si lo deseas.
""")