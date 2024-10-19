import streamlit as st

st.set_page_config(page_title="Seismic Analyzer", layout="wide")

st.title("Seismic Analyzer")

st.write("""
Welcome to the Seismic Analyzer app. This application allows you to record and analyze seismic data.

Please select a page from the sidebar to get started:
- **Recorder**: Connect your mobile device and record seismic data in real-time.
- **Analyzer**: Analyze previously recorded seismic data.
""")