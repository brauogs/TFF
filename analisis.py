import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import hashlib
import os
from datetime import datetime

# Initialize SQLite database
conn = sqlite3.connect('seismic_data.db')
c = conn.cursor()

# Create users table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE NOT NULL,
              password TEXT NOT NULL)''')

# Create data table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS seismic_data
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              filename TEXT,
              data BLOB,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY (user_id) REFERENCES users(id))''')

conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    return c.fetchone() is not None

def save_data(user_id, filename, data):
    c.execute("INSERT INTO seismic_data (user_id, filename, data) VALUES (?, ?, ?)",
              (user_id, filename, data))
    conn.commit()

def load_data(user_id):
    c.execute("SELECT filename, data FROM seismic_data WHERE user_id=?", (user_id,))
    return c.fetchall()

def main():
    st.set_page_config(page_title="Análisis del Acelerograma", layout="wide")

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if st.session_state.user_id is None:
        st.title("Login / Registro")
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Iniciar sesión"):
                if verify_user(username, password):
                    c.execute("SELECT id FROM users WHERE username=?", (username,))
                    st.session_state.user_id = c.fetchone()[0]
                    st.success("Inicio de sesión exitoso!")
                else:
                    st.error("Usuario o contraseña incorrectos")
        with col2:
            if st.button("Registrarse"):
                if create_user(username, password):
                    st.success("Usuario creado exitosamente!")
                else:
                    st.error("El usuario ya existe")
    else:
        st.title("Análisis del Acelerograma")
        
        uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            save_data(st.session_state.user_id, uploaded_file.name, df.to_csv(index=False))
            st.success("Archivo cargado y guardado exitosamente!")
            
            # Your existing data processing and visualization code here
            # ...

        st.sidebar.title("Archivos guardados")
        saved_files = load_data(st.session_state.user_id)
        for filename, data in saved_files:
            if st.sidebar.button(f"Cargar {filename}"):
                df = pd.read_csv(pd.compat.StringIO(data))
                st.write(f"Archivo cargado: {filename}")
                # Your existing data processing and visualization code here
                # ...

        if st.sidebar.button("Cerrar sesión"):
            st.session_state.user_id = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()