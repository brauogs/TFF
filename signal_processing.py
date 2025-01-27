from scipy.signal import savgol_filter

def reducir_ruido(datos, window_length=51, polyorder=3):
    """
    Aplica un filtro Savitzky-Golay para reducir el ruido en los datos
    
    Parámetros:
    - datos: array de numpy con los datos
    - window_length: longitud de la ventana (debe ser impar)
    - polyorder: orden del polinomio
    
    Retorna:
    - datos filtrados
    """
    if window_length % 2 == 0:
        window_length += 1  # Asegurarse que la longitud sea impar
    return savgol_filter(datos, window_length, polyorder)

