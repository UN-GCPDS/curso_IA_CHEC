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
    Nos enfocaremos en explorar el dataset usando las funciones clave como `.head()`, `.info()`, y `.value_counts()`, entre 
    otras. Utilizaremos la siguiente base de datos [Electric Power Consumption](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption)
    """)

    st.markdown("**Objetivo del Módulo:** Comprender el uso de funciones básicas de exploración de datos en `pandas`.")

    st.markdown("""**Descripción de la base de datos:** Este dataset contiene datos sobre el consumo de energía en la 
    ciudad de Tetuán, ubicada en el norte de Marruecos. Se centra en el análisis de cómo varios factores climáticos y otros 
    parámetros afectan el consumo de energía en tres zonas diferentes de la ciudad debido a que Tetúan está ubicada a lo largo del mar Mediterráneo, 
    con un clima suave y lluvioso en invierno, y caluroso y seco en verano.""")


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

# sección indexacción básica

with st.container():
    st.header("4. Indexación básica")
    st.write("""
    Recuerda que podemos elegir solo ver algunas filas o columnas dependiendo de la tarea en la que estemos interesados en ese momento.
    Por ejemplo, selecciona aquellas columnas que nos aporten únicamente el consumo de energía y el resgitro del tiempo en el cual fue tomada la muestra.
    """)


    # Crear checkboxes para seleccionar columnas
    # Los checkboxes devolverán True o False dependiendo de si se han marcado
    columnas_seleccionadas = []
    for col in consumption.columns:
        if st.checkbox(col):
            columnas_seleccionadas.append(col)

    columnas_interes = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3', 'Datetime']


    # Botón para actualizar la selección
    if st.button("Actualizar selección"):
        # Verificar si el usuario ha seleccionado exactamente las últimas tres columnas
        if set(columnas_seleccionadas) == set(columnas_interes):
            st.success("¡Muy bien! Has seleccionado las  4 columnas que corresponden al consumo de energía.")
            # Mostrar DataFrame filtrado con las últimas  columnas seleccionadas
            st.dataframe(consumption[columnas_seleccionadas])
        else:
            st.error("No seleccionaste las columnas correctamente. Recuerda que, en este caso, las columnas que tienen datos sobre el consumo de energía son aquellas que tienen 'PowerConsumption' en su nombre.")
            st.dataframe(consumption[columnas_seleccionadas])

    st.write("Ahora, escribe el índice de la fila de la cual te gustaría conocer sus consumos de energía")

    input_index = st.number_input("Escribe el número del índice:", min_value=0, max_value = len(consumption), step=1)
    st.dataframe(consumption.loc[[input_index], [consumption.columns[i] for i in [0,-3,-2,-1]]])

    st.write("Ahora practiquemos, ¿cuál es el valor de PowerConsumption_Zone2 para la fila que tiene índice 15342?")

    # Input para el número esperado de valores faltantes
    valor = st.number_input("Introduce el valor:", min_value=0.0, step=0.001, format="%.3f")
    valor_expected = consumption.loc[[15342],['PowerConsumption_Zone2']].value[0]

    # Verificar si el valor ingresado es correcto
    if valor:
        st.write(f"{valor}, {valor_expected}")
        if np.round(valor_expected, 3) == np.round(valor,3):
            st.success(f"Muy bien, el valor de PowerConsumption_Zone2 para la fila que tiene índice 15342 es {valor_expected}")
        else:
            # Mostrar el número real de valores faltantes
            st.write(f"El valor total es incorrecto. Recuerda que puedes buscar la fila por su índice.")




    # # Botón para actualizar la selección
    # if st.button("Actualizar selección"):
    #     # Verificar si hay columnas seleccionadas
    #     if columnas_seleccionadas:
    #         # Mostrar DataFrame filtrado
    #         st.write("Mostrando las columnas seleccionadas:")
    #         st.dataframe(consumption[columnas_seleccionadas])
    #     else:
    #         st.write("Selecciona al menos una columna para mostrar el DataFrame.")


# Mensaje de cierre del módulo
st.write("¡Fin del módulo! Ahora ya sabes cómo hacer una exploración inicial de datasets en `pandas`.")