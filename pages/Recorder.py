import streamlit as st
import qrcode
import io
import base64
import uuid
import asyncio
import websockets
import json

st.set_page_config(page_title="Seismic Recorder", layout="wide")

st.title("Seismic Recorder")

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

if 'device_id' not in st.session_state:
    st.session_state.device_id, qr_code = generate_qr_code()
    st.image(qr_code, caption="Scan this QR code to connect your device")

st.write(f"Device ID: {st.session_state.device_id}")

if 'connected' not in st.session_state:
    st.session_state.connected = False

if not st.session_state.connected:
    st.warning("Waiting for device connection...")
else:
    st.success("Device connected!")
    if st.button("Test Connection"):
        # Here you would send a test message to the connected device
        st.success("Test message sent to device")

    st.subheader("Real-time Accelerometer Data")
    chart = st.line_chart()

    # Here you would update the chart with real-time data from the WebSocket connection

# WebSocket server code would go here
# You'd need to set up a WebSocket server to handle connections and data streaming