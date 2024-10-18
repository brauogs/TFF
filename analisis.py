import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import qrcode
import io
import base64
import uuid

def generate_qr_code():
    device_id = str(uuid.uuid4())
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(device_id)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return device_id, f"data:image/png;base64,{img_str}"

def main():
    st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

    st.title("Análisis del Acelerograma")

    if 'device_id' not in st.session_state:
        st.session_state.device_id, qr_code = generate_qr_code()
        st.image(qr_code, caption="Escanea este código QR para conectar tu dispositivo")

    st.write(f"ID del dispositivo: {st.session_state.device_id}")

    uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Archivo cargado exitosamente!")
        
        process_and_visualize_data(df)

def process_and_visualize_data(df):
    st.subheader("Datos del Acelerómetro")
    st.write(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['x'], mode='lines', name='X'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['y'], mode='lines', name='Y'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['z'], mode='lines', name='Z'))
    fig.update_layout(title="Datos del Acelerómetro", xaxis_title="Tiempo", yaxis_title="Aceleración (cm/s²)")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()