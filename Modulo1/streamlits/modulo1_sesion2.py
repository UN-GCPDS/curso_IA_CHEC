# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import warnings  # Eliminar warnings
from sklearn.datasets import fetch_california_housing
warnings.filterwarnings(action="ignore", message="^internal gelsd")
import io
import seaborn as sns
import matplotlib.pyplot as plt

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
    **Sesión 2:** en esta sesión aprenderemos a visualizar datos usando Matplotlib, Seaborn y Plotly. Nos enfocaremos en las 
    técnicas fundamentales para explorar el dataset mediante visualizaciones, como histogramas, gráficos de densidad, scatters, 
    gráficos de pastel, boxplots, diagramas de violín y geopandas. Exploraremos cómo cada una de estas bibliotecas nos permite
     analizar y comprender mejor la distribución y relaciones entre las variables clave. Utilizaremos la siguiente base de datos
      [Electric Power Consumption](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption)
    """)

    st.markdown("""**Descripción de la base de datos:** Este dataset contiene datos sobre el consumo de energía en la 
    ciudad de Tetuán, ubicada en el norte de Marruecos. Se centra en el análisis de cómo varios factores climáticos y otros 
    parámetros afectan el consumo de energía en tres zonas diferentes de la ciudad debido a que Tetúan está ubicada a lo largo del mar Mediterráneo, 
    con un clima suave y lluvioso en invierno, y caluroso y seco en verano. A continuación se hace una pequeña descripción de
    cada variable (columna):""")

    st.markdown("""
    - **Date Time**: Ventana de tiempo de diez minutos.
    - **Temperature**: Temperatura del clima.
    - **Humidity**: Humedad del clima.
    - **Wind Speed**: Velocidad del viento.
    - **General Diffuse Flows**: El término "flujo difuso" describe fluidos de baja temperatura (< 0.2° a ~ 100°C) que se descargan lentamente a través de montículos de sulfuro, flujos de lava fracturados y ensamblajes de tapetes bacterianos y macrofauna.
    - **Diffuse Flows**
    - **Zone 1 Power Consumption**: Consumo de energía en la Zona 1.
    - **Zone 2 Power Consumption**: Consumo de energía en la Zona 2.
    - **Zone 3 Power Consumption**: Consumo de energía en la Zona 3.
    """)

# Sección: Observación del Dataset con .head()
with st.container():
    st.header("1. Exploración Inicial del Dataset con `.head()`")


    num_filas = st.number_input('Selecciona el número de filas a mostrar:', min_value=1, max_value=50, step=1, value=5)
    st.dataframe(consumption.head(num_filas))

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>consumption.head({num_filas})<\code>
    </div>
    """, unsafe_allow_html=True)

# Sección: Histograma Personalizado
with st.container():
    st.header("2. Visualización de Histograma con `.hist()`")

    # Seleccionar la columna para el histograma
    columnas_numericas = consumption.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    columna_seleccionada = st.selectbox('Selecciona la columna para el histograma:', columnas_numericas)

    # Seleccionar el número de bins
    bins = st.slider('Selecciona el número de bins para el histograma:', min_value=5, max_value=100, step=1, value=20)

    # Mostrar el histograma
    fig, ax = plt.subplots()
    ax.hist(consumption[columna_seleccionada], bins=bins, color='skyblue', edgecolor='black')
    ax.set_title(f'Histograma de {columna_seleccionada}')
    ax.set_xlabel(columna_seleccionada)
    ax.set_ylabel('Frecuencia')
    
    # Mostrar la gráfica en la app
    st.pyplot(fig)

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>plt.hist({columna_seleccionada}, bins={bins})<\code></small>
    </div>
    """, unsafe_allow_html=True)


