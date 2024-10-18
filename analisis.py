import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import qrcode
import io
import base64
import uuid
import websockets
import asyncio
import json
from datetime import datetime

# Global variables
connected_devices = {}

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

async def handle_websocket(websocket, path):
    device_id = path.split('/')[-1]
    connected_devices[device_id] = websocket
    try:
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'accelerometer_data':
                # Process and store the accelerometer data
                # You can implement real-time data visualization here
                pass
    finally:
        del connected_devices[device_id]

def main():
    st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

    st.title("Análisis del Acelerograma")

    if 'device_id' not in st.session_state:
        st.session_state.device_id, qr_code = generate_qr_code()
        st.image(qr_code, caption="Escanea este código QR para conectar tu dispositivo")

    if st.session_state.device_id in connected_devices:
        st.success("Dispositivo conectado")
        if st.button("Probar conexión"):
            asyncio.run(connected_devices[st.session_state.device_id].send(json.dumps({'type': 'test_connection'})))
            st.success("Señal de prueba enviada al dispositivo")

        # Real-time data visualization
        st.subheader("Datos del Acelerómetro en Tiempo Real")
        chart = st.line_chart(pd.DataFrame(columns=['x', 'y', 'z']))

        # You can update this chart with real-time data from the WebSocket connection

    else:
        st.warning("Esperando conexión del dispositivo...")

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
    # Start WebSocket server
    start_server = websockets.serve(handle_websocket, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()