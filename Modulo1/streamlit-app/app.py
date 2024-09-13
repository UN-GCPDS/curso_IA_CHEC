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
    df = pd.read_csv('datos/california_housing_train.csv')
    return df
# Fijar semilla para fines pedagógicos
np.random.seed(42)

# Cargar el dataset de precios de viviendas en California
housing = load_data()

# Título del módulo
st.title("Módulo 1: Programación y Estadística Básica con Python")

# Sección introductoria
with st.container():
    st.subheader("Introducción al Módulo")
    st.write("""
    En este módulo aprenderemos a manejar datos con `pandas`, utilizando un dataset de precios de vivienda en California.
    Nos enfocaremos en explorar el dataset usando las funciones clave como `.head()`, `.info()`, y `.value_counts()`, entre otras.
    """)

    st.markdown("**Objetivo del Módulo:** Comprender el uso de funciones básicas de exploración de datos en `pandas`.")

# Sección: Importación de librerías y datos
with st.container():
    st.header("1. Importación de Librerías y Dataset")
    st.write("Primero, vamos a asegurarnos de que las librerías y el dataset estén correctamente cargados. Este es el código que utilizamos para importarlos:")

    librerias_code = """
    import pandas as pd  # Manipulación de datos tabulares
    housing = pd.read_csv('/content/sample_data/california_housing_train.csv')  # Cargar dataset
    """
    st.code(librerias_code, language="python")

# Sección: Observación del Dataset con .head()
with st.container():
    st.header("2. Exploración Inicial del Dataset con `.head()`")
    st.write("""
    La función `.head()` nos permite visualizar las primeras filas del dataset, facilitando una primera exploración rápida y general de la estructura y el contenido de los datos.
    Esto es particularmente útil para identificar la naturaleza de las columnas y la calidad de los datos al comenzar un análisis.
    """)

    head_code = "housing.head(5)"
    st.code(head_code, language='python')

    st.write("""
    Puedes ajustar el número 5 a cualquier número entero positivo para mostrar más o menos filas según sea necesario.
    A continuación, puedes seleccionar cuántas filas deseas visualizar en la tabla.
    """)

    num_filas = st.number_input('Selecciona el número de filas a mostrar:', min_value=1, max_value=50, step=1, value=5)
    st.dataframe(housing.head(num_filas))


# Sección: Información del Dataset con .info()
with st.container():
    st.header("3. Información General del Dataset con `.info()`")
    st.write("""
    La función `.info()` nos da un resumen del dataset, incluyendo el número de entradas, los tipos de datos de cada columna y la cantidad de valores nulos.
    Esto es muy útil para entender la estructura de los datos y posibles problemas de calidad (como valores faltantes).
    """)

    info_code = "housing.info()"
    st.code(info_code)

    # Captura la salida de housing.info()
    buffer = io.StringIO()
    housing.info(buf=buffer)
    info_str = buffer.getvalue()

    # Mostrar el texto en Streamlit
    st.write("Obteniendo como salida: ")
    st.text(info_str)


# Sección: Conteo de Valores con .value_counts()
with st.container():
    st.header("4. Análisis de Frecuencia con `.value_counts()`")
    st.write("""
    La función `.value_counts()` es útil para analizar la frecuencia de los valores en una columna específica. Por ejemplo, podemos ver cuántas veces
    aparece cada valor en una columna categórica o discreta. Por ejemplo, a continuación se presenta cómo hacer el análisis sobre total_rooms:
    """)

    st.code('housing["total_rooms"].value_counts()')

    st.write("Obteniendo como salida")
    st.dataframe(housing["total_rooms"].value_counts())

    st.write("""
    Puedes modificar el argumento "total_rooms" a cualquier columna.
    A continuación, puedes seleccionar una columna para analizar la frecuencia de sus valores.
    """)

    # Selección de columna para aplicar .value_counts()
    columna_seleccionada = st.selectbox(
        "Selecciona una columna:",
        options=['housing_median_age', 'total_rooms',
       'total_bedrooms', 'households', 'median_house_value']
    )

    # Mostrar los resultados de .value_counts()
    st.subheader(f"Frecuencia de valores en la columna '{columna_seleccionada}'")
    st.write(housing[columna_seleccionada].value_counts())

# Mensaje de cierre del módulo
st.write("¡Fin del módulo! Ahora ya sabes cómo hacer una exploración inicial de datasets en `pandas`.")