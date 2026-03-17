import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo de regresión
regressor = load('gradientboosting_model_reg.joblib')

# Cargar el encoder
#with open('encoderpipeline.pickle', 'rb') as f:
#    encoder = pickle.load(f)

# Inicializar variables
#rd_spend = administration = marketing_spend = 0.0
#selected_state = "New York"

pct_acum_08 = pct_acum_07 = pct_acum_06 = pct_acum_05 = mto_pia = mto_pim = 0.0

var_pct_08 = var_pct_07 = var_pct_06 = ratio_pia_pim = 0.0

# Streamlit app
st.title("Modelo de Regresión Gradient Boosting")
st.markdown("##### Si colocas un valor negativo, aparecerá un error y no podrás completar otros campos. La predicción será incorrecta.")

# Sidebar para la entrada del usuario
st.sidebar.header("Campos a Evaluar")

# Entrada del usuario para mto_pia
mto_pia = st.sidebar.number_input("**Monto PIA (Min=0, Max=19185702)**", min_value=0.0, value=float(mto_pia))

# Entrada del usuario para mto_pim
mto_pim = st.sidebar.number_input("**Monto PIM (Min=0, Max=13280780)**", min_value=0.0, value=float(mto_pim))

# Entrada del usuario para pct_acum_05
pct_acum_05 = st.sidebar.number_input("**Ejec acum May(%) (Min=0, Max=1)**", min_value=0.0, value=float(pct_acum_05))

# Entrada del usuario para pct_acum_05
pct_acum_06 = st.sidebar.number_input("**Ejec acum Jun(%) (Min=0, Max=1)**", min_value=0.0, value=float(pct_acum_06))

# Entrada del usuario para pct_acum_05
pct_acum_07 = st.sidebar.number_input("**Ejec acum Jul(%) (Min=0, Max=1)**", min_value=0.0, value=float(pct_acum_07))

# Entrada del usuario para pct_acum_05
pct_acum_08 = st.sidebar.number_input("**Ejec acum Ago(%) (Min=0, Max=1)**", min_value=0.0, value=float(pct_acum_08))


# Calculo de las variables del modelo
var_pct_08 = pct_acum_08 - pct_acum_07
var_pct_07 = pct_acum_07 - pct_acum_06
var_pct_06 = pct_acum_06 - pct_acum_05
ratio_pia_pim = mto_pia/mto_pim

# Función para resetear las entradas
def reset_inputs():
    global pct_acum_08, pct_acum_07, pct_acum_06, pct_acum_05, mto_pia, mto_pim
    pct_acum_08 = pct_acum_07 = pct_acum_06 = pct_acum_05 = mto_pia = mto_pim = 0.0

# Botón para predecir
if st.sidebar.button("Predecir"):
    # Validar las entradas
    if all(isinstance(val, (int, float)) and val >= 0 for val in [pct_acum_08, var_pct_08, var_pct_07, var_pct_06, ratio_pia_pim]):
        # Crear un DataFrame con las entradas del usuario
        obs = pd.DataFrame({
            'pct_acum_08': [pct_acum_08],
            'var_pct_08': [var_pct_08],
            'var_pct_07': [var_pct_07],
            'var_pct_06': [var_pct_06],
            'ratio_pia_pim': [ratio_pia_pim]
        })

        # Mostrar el DataFrame de entradas para depuración
        st.write("DataFrame de Entradas:")
        st.write(obs)

        #----------------------Pipeline-------------------------
        # Predecir usando el modelo
        target = regressor.predict(obs)

        # Mostrar la predicción con un tamaño de fuente grande usando markdown
        st.markdown(f'<p style="font-size: 40px; color: green;">La predicción del porcentaje acumulado de ejecución de Setiembre será: ${target[0]:,.2f}</p>', unsafe_allow_html=True)

    else:
        st.warning("Rellene todos los espacios en blanco")

# Colocar el botón "Resetear" debajo del botón "Predecir"
if st.sidebar.button("Resetear"):
    # Resetear inputs
    reset_inputs()