import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo de regresión
regressor = load('extremegradientboosting_model_reg.joblib')

# Inicializar variables
mto_dev_08 = mto_dev_07 = mto_dev_01_06 = mto_pim = 0.0
select_departamento_meta = "15.LIMA"
select_categoria_gasto = "6.GASTOS DE CAPITAL"

# Streamlit app
st.set_page_config(layout="wide")
st.title("Modelo de Regresión eXtreme Gradient Boosting")
st.markdown("##### Si colocas un valor negativo, aparecerá un error y no podrás completar otros campos. La predicción será incorrecta.")

# Sidebar para la entrada del usuario
st.sidebar.header("Campos a Evaluar")

# Entrada del usuario para monto de PIM
mto_pim = st.sidebar.number_input("**Monto PIM (Min=0)**", min_value=0.0, value=float(mto_pim))

# Entrada del usuario para monto devengado acumulado al primer semestre
mto_dev_01_06 = st.sidebar.number_input("**Monto devengado al 1er Semestre (Min=0)**", min_value=0.0, value=float(mto_dev_01_06))

# Entrada del usuario para monto devengado en el mes de julio
mto_dev_07 = st.sidebar.number_input("**Monto devengado en Julio (Min=0)**", min_value=0.0, value=float(mto_dev_07))

# Entrada del usuario para monto devengado en el mes de agosto
mto_dev_08 = st.sidebar.number_input("**Monto devengado en Agosto (Min=0)**", min_value=0.0, value=float(mto_dev_08))

# Entrada del usuario para el nombre del departamento meta
st.sidebar.markdown("<h1 style='font-size: 24px;'>Región de ejecución</h1>", unsafe_allow_html=True)
select_departamento_meta = st.sidebar.selectbox("Selecciona Región", [
    "AREQUIPA",
    "CAJAMARCA",
    "CALLAO",
    "CUSCO",
    "HUANCAVELICA",
    "HUANUCO",
    "ICA",
    "JUNIN",
    "LA LIBERTAD",
    "LIMA",
    "LORETO",
    "MADRE DE DIOS",
    "MOQUEGUA",
    "PASCO",
    "PIURA",
    "PUNO",
    "SAN MARTIN",
    "UCAYALI",
    "OTROS_DEPARTAMENTOS"
    ], index=[
        "04.AREQUIPA",
        "06.CAJAMARCA",
        "07.PROVINCIA CONSTITUCIONAL DEL CALLAO",
        "08.CUSCO",
        "09.HUANCAVELICA",
        "10.HUANUCO",
        "11.ICA",
        "12.JUNIN",
        "13.LA LIBERTAD",
        "15.LIMA",
        "16.LORETO",
        "17.MADRE DE DIOS",
        "18.MOQUEGUA",
        "19.PASCO",
        "20.PIURA",
        "21.PUNO",
        "22.SAN MARTIN",
        "25.UCAYALI",
        "OTROS_DEPARTAMENTOS"
    ].index(select_departamento_meta))


# Entrada del usuario para el nombre de la categoria de gasto
st.sidebar.markdown("<h1 style='font-size: 24px;'>Categoria de gasto</h1>", unsafe_allow_html=True)
select_categoria_gasto = st.sidebar.selectbox("Selecciona Categoria", ["GASTOS CORRIENTES","GASTOS DE CAPITAL"], 
                                              index=["5.GASTOS CORRIENTES","6.GASTOS DE CAPITAL"].index(select_categoria_gasto))

# Calculo de las variables del modelo
pct_acum_08 = pct_acum_07 = var_pct_08 = 0.0

if mto_pim > 0:
    pct_acum_08 = (mto_dev_01_06 + mto_dev_07 + mto_dev_08) / mto_pim
    pct_acum_07 = (mto_dev_01_06 + mto_dev_07) / mto_pim
    var_pct_08 = pct_acum_08 - pct_acum_07
else:
    pct_acum_08 = pct_acum_07 = var_pct_08 = 0.0

# Función para resetear las entradas
def reset_inputs():
    global mto_dev_01_06, mto_dev_07, mto_dev_08, mto_pim, select_departamento_meta, select_categoria_gasto
    mto_dev_08 = mto_dev_07 = mto_dev_01_06 = mto_pim = 0.0
    select_departamento_meta = "15.LIMA"
    select_categoria_gasto = "6.GASTOS DE CAPITAL"

# Botón para predecir
if st.sidebar.button("Predecir"):
    # Validar las entradas
    if all(isinstance(val, (int, float)) and val >= 0 for val in [mto_dev_01_06, mto_dev_07, mto_dev_08, mto_pim]):
        # Crear un DataFrame con las entradas del usuario
        entrada = pd.DataFrame({
            'Monto PIM': [mto_pim],
            'Ejecutado al 1er Semestre': [mto_dev_01_06],
            'Ejecutado en Julio': [mto_dev_07],
            'Ejecutado en Agosto': [mto_dev_08],
            'Región Meta': [select_departamento_meta],
            'Categoria Gasto': [select_categoria_gasto]
        })
        
        obs = pd.DataFrame({
            'pct_acum_08': [pct_acum_08],
            'var_pct_08': [var_pct_08],
            'departamento_meta': [select_departamento_meta],
            'categoria_gasto': [select_categoria_gasto]
        })

        # Mostrar el DataFrame de entradas para depuración
        st.write("DataFrame de Entradas para la aplicación:")
        st.write(entrada)
        st.write("DataFrame de Entradas para el modelo:")
        st.write(obs)

        #----------------------Pipeline-------------------------
        # Predecir usando el modelo
        target = regressor.predict(obs)

        # Mostrar la predicción con un tamaño de fuente grande usando markdown
        st.markdown(f'<p style="font-size: 40px; color: green;">Al mes de setiembre se predice ejecutar S/.{target[0]*mto_pim:.2f} que representa el {target[0]*100:.2f}% del PIM</p>', unsafe_allow_html=True)

    else:
        st.warning("Rellene todos los espacios en blanco")

# Colocar el botón "Resetear" debajo del botón "Predecir"
if st.sidebar.button("Resetear"):
    # Resetear inputs
    reset_inputs()
