# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import warnings  # Eliminar warnings
from sklearn.datasets import fetch_california_housing
warnings.filterwarnings(action="ignore", message="^internal gelsd")
import io

@st.cache_data
def load_data():
    # Fetch the dataset as a pandas DataFrame
    df = pd.read_csv('Modulo1/databases/powerconsumption.csv')
    return df

# Fijar semilla para fines pedagógicos
np.random.seed(42)

# Cargar el dataset de precios de viviendas en California
consumption = load_data()

np.random.seed(42)
num_deleted = 50
filas = np.random.randint(len(consumption), size=num_deleted)
columnas = np.random.randint(len(consumption.columns), size=num_deleted)

# Reemplazar celdas específicas con NaN
for i in range(len(filas)):
    fila_index = filas[i]
    columna_index = columnas[i]
    consumption.iloc[fila_index, columna_index] = None

# Título del módulo
st.title("Módulo 1: Programación y Estadística Básica con Python")

# Sección introductoria
with st.container():
    st.subheader("Introducción al Módulo")
    st.write("""
    En este módulo aprenderemos a manejar datos con `pandas`, utilizando un dataset de precios de vivienda en California.
    Nos enfocaremos en explorar el dataset usando las funciones clave como `.head()`, `.info()`, y `.value_counts()`, entre otras. Con la base
    de datos [Electric Power Consumption](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption)
    """)

    st.markdown("**Objetivo del Módulo:** Comprender el uso de funciones básicas de exploración de datos en `pandas`.")


# Sección: Observación del Dataset con .head()
with st.container():
    st.header("1. Exploración Inicial del Dataset con `.head()`")


    num_filas = st.number_input('Selecciona el número de filas a mostrar:', min_value=1, max_value=50, step=1, value=5)
    st.dataframe(consumption.head(num_filas))


# Sección: Información del Dataset con .info()
with st.container():
    st.header("2. Información General del Dataset con `.info()`")
    st.write("""
    La función `.info()` nos da un resumen del dataset, incluyendo el número de entradas, los tipos de datos de cada columna y la cantidad de valores nulos.
    Esto es muy útil para entender la estructura de los datos y posibles problemas de calidad (como valores faltantes).
    """)

    # Captura la salida de housing.info()
    buffer = io.StringIO()
    consumption.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str) # text

    st.write(f"Hagamos un pequeño análisis.")

    # pregunta por valores totales

    st.write("¿Cuál es el valor total de entradas (filas) de la base de datos?")

    # Input para el número esperado de valores faltantes
    expected_total = st.number_input("Introduce el número que crees que es el total:", min_value=0, step=1)
    total = len(consumption)

    # Verificar si el valor ingresado es correcto
    if expected_total:
        if expected_total == total:
            st.success(f"Muy bien, el valor total de entradas es {total}")
        else:
            # Mostrar el número real de valores faltantes
            st.write(f"El valor total es incorrecto. Recuerda que las entradas totales se definen como 'entries'")


    # pregunta de valores faltantes

    col_name = consumption.columns[2]



    st.write(f"En la columna '{col_name}', ¿cuántos valores faltantes hay?")

    # Input para el número esperado de valores faltantes
    expected_missing = st.number_input("Introduce el número que crees que son los valores faltantes:", min_value=0, step=1)

    # Contar valores faltantes en la columna
    missing_values = consumption[col_name].isnull().sum()

    # Verificar si el valor ingresado es correcto
    if expected_missing:
        if missing_values == expected_missing:
            st.success(f"Muy bien, los valores faltantes en la columna '{col_name}' son {missing_values}")
        else:
            # Mostrar el número real de valores faltantes
            st.write(f"El número de valores faltantes es incorrecto. Recuerda que debes tomar el valor total de valores y restarle la cantidad de no nulos de la columna '{col_name}'")



# Sección: Conteo de Valores con .value_counts()
with st.container():
    st.header("3. Análisis de Frecuencia con `.value_counts()`")
    st.write("""
    La función `.value_counts()` es útil para analizar la frecuencia de los valores en una columna específica. Por ejemplo, podemos ver cuántas veces
    aparece cada valor en una columna categórica o discreta. Por ejemplo, a continuación se presenta cómo hacer el análisis sobre Temperature:
    """)

    st.dataframe(consumption["Temperature"].value_counts())

    st.write("""
    Puedes modificar el argumento "Temperature" a cualquier columna.
    A continuación, puedes seleccionar una columna para analizar la frecuencia de sus valores.
    """)

    # Selección de columna para aplicar .value_counts()
    columna_seleccionada = st.selectbox(
        "Selecciona una columna:",
        options=consumption.columns
    )

    # Mostrar los resultados de .value_counts()
    st.subheader(f"Frecuencia de valores en la columna '{columna_seleccionada}'")
    st.write(consumption[columna_seleccionada].value_counts())

    # Pregunta

    st.write(f"¿Cuál es el valor más frecuente de la columna DiffuseFlows?")

    # Input para el número esperado de valores faltantes
    freq = st.number_input("Introduce el valor más frecuente:", min_value=0.0, step=0.001, format="%.3f")
    freq_expected = consumption['DiffuseFlows'].value_counts().index[0]

    # Verificar si el valor ingresado es correcto
    if freq:
        if freq_expected == freq:
            st.success(f"Muy bien, el valor más frecuente de la columna DiffuseFlows	 es {freq_expected}")
        else:
            # Mostrar el número real de valores faltantes
            st.write(f"El valor total es incorrecto. Recuerda es el primer valor que obtenemos en la tabla.")

# Mensaje de cierre del módulo
st.write("¡Fin del módulo! Ahora ya sabes cómo hacer una exploración inicial de datasets en `pandas`.")