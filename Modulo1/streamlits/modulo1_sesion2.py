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
import plotly.express as px

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
    **Sesión 2:** en esta sesión aprenderemos a usar modelos estadísticos clásicos de análisis exploratorio datos usando librerías como Matplotlib, Seaborn y Plotly. Nos enfocaremos en las 
    técnicas fundamentales para explorar el dataset mediante visualizaciones, como histogramas, gráficos de densidad, scatters, 
    gráficos de pastel, boxplots, diagramas de violín y geopandas. Exploraremos cómo cada una de estas bibliotecas nos permite
     analizar y comprender mejor la distribución y relaciones entre las variables clave. A partir de la sección 4, utilizaremos la base de datos
      [Electric Power Consumption](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption).
    """)

# Sección: Probabilidad - Reglas de Suma y Multiplicación
with st.container():
    st.header("1. Reglas de Suma y Multiplicación en Probabilidades")

    st.markdown("""
    Las reglas de la **suma** y la **multiplicación** en probabilidades permiten calcular la probabilidad de la ocurrencia de uno o más eventos.
    
    - **Regla de la suma**: Si A y B son eventos mutuamente excluyentes, la probabilidad de que ocurra A o B es la suma de las probabilidades individuales de A y B.
    - **Regla de la multiplicación**: Si A y B son eventos independientes, la probabilidad de que ocurran ambos eventos es el producto de sus probabilidades individuales.
    """)

    # Datos sintéticos: Supongamos que tenemos dos eventos A y B con probabilidades dadas
    p_A = 0.6
    p_B = 0.3
    p_A_and_B = p_A * p_B  # Eventos independientes

    st.write(f"La probabilidad de que ocurra el evento A es {p_A}.")
    st.write(f"La probabilidad de que ocurra el evento B es {p_B}.")
    st.write(f"Si los eventos son independientes, la probabilidad de que ocurran ambos es {p_A_and_B}.")

    # Pregunta interactiva
    st.markdown("### Pregunta:")
    pregunta_suma = st.radio("Si A y B son mutuamente excluyentes, ¿cuál es la probabilidad de que ocurra A o B?", 
                              ['0.9', '0.5', '0.3'])
    
    if st.button("Validar respuesta", key="val_suma"):
        if pregunta_suma == '0.9':
            st.success("¡Correcto! La probabilidad de que ocurra A o B es 0.9.")
        else:
            st.error("Incorrecto. Si A y B son mutuamente excluyentes, la probabilidad de que ocurra A o B es la suma de sus probabilidades individuales.")

# Sección: Funciones de Densidad y Distribución
with st.container():
    st.header("2. Funciones de Densidad y de Distribución")

    st.markdown("""
    Las funciones de densidad y distribución son fundamentales en probabilidad:

    - **Función de Densidad de Probabilidad (PDF)**: Describe la probabilidad de que una variable continua tome un valor específico.
    - **Función de Distribución Acumulativa (CDF)**: Indica la probabilidad de que una variable aleatoria sea menor o igual a un valor dado.
    """)

    # Datos sintéticos: Distribución normal
    mean = 0
    std_dev = 1
    data = np.random.normal(mean, std_dev, 1000)

    # Mostrar la gráfica de la función de densidad
    fig_pdf, ax_pdf = plt.subplots()
    sns.histplot(data, kde=True, stat="density", linewidth=0, ax=ax_pdf)
    ax_pdf.set_title('Función de Densidad de Probabilidad (PDF)')
    st.pyplot(fig_pdf)

    # Pregunta interactiva
    st.markdown("### Pregunta:")
    pregunta_pdf = st.radio("¿Qué característica describe mejor una función de densidad de probabilidad?", 
                            ['Representa probabilidades acumuladas', 'Describe la probabilidad para variables discretas', 'Describe la probabilidad para variables continuas'])
    
    if st.button("Validar respuesta", key="val_pdf"):
        if pregunta_pdf == 'Describe la probabilidad para variables continuas':
            st.success("¡Correcto! La PDF describe la probabilidad para variables continuas.")
        else:
            st.error("Incorrecto. La PDF describe la probabilidad de variables continuas.")

# Sección: Familias Paramétricas Clásicas y Métodos No Paramétricos
with st.container():
    st.header("3. Familias Paramétricas Clásicas y Métodos No Paramétricos")

    st.markdown("""
    - **Familias Paramétricas**: Se basan en suposiciones específicas sobre la distribución de los datos (por ejemplo, distribución normal, binomial).
    - **Métodos No Paramétricos**: No asumen una forma específica para la distribución de los datos, lo que los hace más flexibles.
    """)

    # Datos sintéticos: Comparación entre una distribución normal y un método no paramétrico (Kernel Density Estimate)
    data_normal = np.random.normal(0, 1, 1000)
    data_nonparam = np.random.uniform(-3, 3, 1000)

    # Gráfica comparativa
    fig_parametric, ax_parametric = plt.subplots()

    # Ajustar histograma para normalizar alturas
    sns.histplot(data_normal, kde=True, color='blue', label='Paramétrico (Normal)', stat="density", ax=ax_parametric)

    # Ajustar el ancho de banda (bandwidth) para la KDE
    sns.kdeplot(data_nonparam, color='red', label='No Paramétrico (KDE)', bw_adjust=0.5, ax=ax_parametric)

    ax_parametric.set_title('Comparación: Paramétrico vs No Paramétrico')
    ax_parametric.legend()
    st.pyplot(fig_parametric)

    # Pregunta interactiva
    st.markdown("### Pregunta:")
    pregunta_parametric = st.radio("¿Cuál es una característica clave de los métodos no paramétricos?", 
                                   ['Asumen una distribución específica', 'Son más flexibles que los métodos paramétricos'])
    
    if st.button("Validar respuesta", key="val_parametric"):
        if pregunta_parametric == 'Son más flexibles que los métodos paramétricos':
            st.success("¡Correcto! Los métodos no paramétricos son más flexibles.")
        else:
            st.error("Incorrecto. Los métodos no paramétricos son más flexibles porque no asumen una forma específica para la distribución.")

# Sección: Observación del Dataset con .head()
with st.container():
    st.header("4. Exploración Inicial del Dataset con `.head()`")

    num_filas = st.number_input('Selecciona el número de filas a mostrar:', min_value=1, max_value=50, step=1, value=5)
    st.dataframe(consumption.head(num_filas))

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>consumption.head({num_filas})</code>
    </div>
    """, unsafe_allow_html=True)

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

# Sección: Histograma Personalizado
with st.container():
    st.header("5. Visualización de Histograma con `.hist()`")

 # Descripción del histograma y bins
    st.markdown("""
    **¿Qué es un histograma?**
    
    Un histograma es una representación gráfica que muestra la distribución de una variable numérica. 
    La variable se divide en intervalos, y la altura de cada barra indica la frecuencia de los valores que caen dentro de ese intervalo.

    **¿Qué son los bins?**
    
    Los "bins" son los intervalos o contenedores en los que se agrupan los datos. El número de bins determina cuántas divisiones tendrá el histograma, lo que influye en el nivel de detalle de la gráfica. Un número menor de bins agrupa más valores en cada barra, mientras que un número mayor de bins permite ver variaciones más finas en los datos.
    """)

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
    <small> Salida generada por <code>plt.hist({columna_seleccionada}, bins={bins})</code></small>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta interactiva sobre la columna 'humidity'
    st.markdown("### Pregunta:")
    st.markdown("En el histograma de la columna 'Humidity', ¿qué valores son más frecuentes?")

    respuesta = st.radio("Selecciona una opción:", ['Valores mayores a 50', 'Valores menores a 50'])

    # Comprobación de la respuesta
    valores_menores_50 = consumption[consumption['Humidity'] < 50]['Humidity'].count()
    valores_mayores_50 = consumption[consumption['Humidity'] >= 50]['Humidity'].count()

    if st.button("Validar respuesta", key="val_hist"):
        if respuesta == 'Valores mayores a 50':
                st.success("¡Correcto! Los valores mayores a 50 son más frecuentes.")
        else:
                st.error("Incorrecto. Observa en qué parte del histograma se encuentran las barras más altas.")

# Sección: Histograma con Seaborn y KDE
with st.container():
    st.header("6. Visualización de Histograma con `sns.histplot` y KDE")

    # Descripción del histograma con KDE
    st.markdown("""
    **¿Qué es un histograma con KDE?**
    
    Un histograma con **KDE (Kernel Density Estimate)** añade una línea suave que estima la densidad de los datos. 
    La curva KDE ayuda a ver mejor la tendencia general de la distribución.

    Existen diferentes tipos de distribuciones, como:

    - Distribución unimodal (la curva tiene un solo pico, lo que sugiere que los datos se agrupan alrededor de un único valor)
    - Distribución bimodal (la curva tiene dos picos, lo que sugiere dos agrupaciones distintas de los datos)
    - Distribución uniforme (la curva es relativamente plana, lo que sugiere que los valores están distribuidos de manera uniforme)
    - Distribución sesgada a la derecha (la curva tiene un pico más hacia la izquierda y una "cola" larga hacia la derecha, lo que sugiere que hay más valores bajos)
    - Distribución sesgada a la izquierda (la curva tiene un pico más hacia la derecha y una "cola" larga hacia la izquierda, lo que sugiere que hay más valores altos)
    """)

    # Seleccionar la columna para el histograma
    columna_seleccionada = st.selectbox('Selecciona la columna para el histograma con KDE:', columnas_numericas)

    # Seleccionar el número de bins
    bins = st.slider('Selecciona el número de bins para el histograma con KDE:', min_value=5, max_value=100, step=1, value=20)

    # Mostrar el histograma con sns.histplot y KDE
    fig, ax = plt.subplots()
    sns.histplot(consumption[columna_seleccionada], bins=bins, kde=True, color='blue', ax=ax)
    ax.set_title(f'Histograma de {columna_seleccionada} con KDE')
    ax.set_xlabel(columna_seleccionada)
    ax.set_ylabel('Frecuencia')

    # Mostrar la gráfica en la app
    st.pyplot(fig)

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>sns.histplot({columna_seleccionada}, bins={bins}, kde=True)</code></small>
    </div>
    """, unsafe_allow_html=True)

    # Sección: Gráfica KDE pura
    st.markdown("### Solo la curva KDE")

    # Generar la gráfica KDE
    fig_kde, ax_kde = plt.subplots()
    kdeplot = sns.kdeplot(consumption[columna_seleccionada], bw_adjust=1, ax=ax_kde)
    ax_kde.set_title(f'Curva KDE de {columna_seleccionada}')
    ax_kde.set_xlabel(columna_seleccionada)
    ax_kde.set_ylabel('Densidad')

    # Mostrar la gráfica KDE pura en la app
    st.pyplot(fig_kde)

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>sns.kdeplot(consumption[{columna_seleccionada}])</code></small>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    Además del histograma, también podemos visualizar solo la curva que estima la densidad de los datos (KDE). 
    Esta curva nos permite observar la forma de la distribución de los valores de una manera más suave, sin la segmentación en bins que tiene el histograma. 
    Es especialmente útil para identificar tendencias generales, como si los datos se agrupan en uno o varios picos, o si están distribuidos de forma más uniforme.
    """)

# Sección: Pregunta interactiva sobre la distribución
st.markdown("### Pregunta:")
st.markdown(f"Observa el histograma con KDE para la columna 'Temperature'. ¿Cómo describirías la distribución de los datos?")

# Opciones sin seleccionar ninguna por defecto
opciones = ['Distribución unimodal', 'Distribución bimodal', 'Distribución uniforme', 'Distribución sesgada a la derecha', 'Distribución sesgada a la izquierda']

# Crear el radio button sin selección predeterminada (index=None no es soportado)
respuesta_distribucion = st.radio("Selecciona una opción:", opciones)

# Botón para confirmar la respuesta
if st.button("Validar respuesta", key="val_dist"):
    # Supongamos que la temperatura tiene una distribución bimodal
    if respuesta_distribucion == 'Distribución bimodal':
        st.success("¡Correcto! La distribución de 'Temperature' es bimodal ya que tiene dos picos claramente visibles: uno alrededor de 15°C y otro alrededor de 20°C.")
    else:
        st.error("Incorrecto. Observa que la distribución de 'Temperature' tiene dos picos.")


# Sección: Gráfico de Dispersión (Scatter plot) con Matplotlib
with st.container():
    st.header("7. Gráfico de Dispersión con `plt.scatter`")

    # Descripción del gráfico de dispersión
    st.markdown("""
        
    Un gráfico de dispersión es una representación gráfica que muestra la relación entre dos variables numéricas.
    Cada punto en el gráfico representa un par de valores de las dos variables seleccionadas. 
    Es útil para identificar correlaciones, patrones o tendencias entre las variables, como una relación positiva, negativa o ninguna relación.

    - **Relación positiva**: Cuando los valores de ambas variables aumentan.
    - **Relación negativa**: Cuando los valores de una variable aumentan mientras que los de la otra disminuyen.
    - **Sin relación aparente**: Los puntos están dispersos sin un patrón claro.
    """)

    # Seleccionar las dos columnas para el scatter plot
    columna_x = st.selectbox('Selecciona la columna para el eje X:', columnas_numericas)
    columna_y = st.selectbox('Selecciona la columna para el eje Y:', columnas_numericas)

    # Mostrar el scatter plot
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(consumption.head(10000)[columna_x], consumption.head(10000)[columna_y], color='blue', alpha=0.6)
    ax_scatter.set_title(f'Dispersión de {columna_x} vs {columna_y}')
    ax_scatter.set_xlabel(columna_x)
    ax_scatter.set_ylabel(columna_y)

    # Mostrar la gráfica en la app
    st.pyplot(fig_scatter)

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>plt.scatter({columna_x}, {columna_y})</code></small>
    </div>
    """, unsafe_allow_html=True)

#Pregunta interactiva sobre el gráfico de dispersión
st.markdown("### Pregunta:")
st.markdown(f"Observa el gráfico de dispersión de 'PowerConsumption_Zone1' (en el eje x) vs 'PowerConsumption_Zone2'. ¿Cómo describirías la relación entre las dos variables?")

# Opciones sin seleccionar ninguna por defecto
opciones_scatter = ['Relación positiva', 'Relación negativa', 'Sin relación aparente']

# Crear el radio button sin selección predeterminada
respuesta_scatter = st.radio("Selecciona una opción:", opciones_scatter)

# Botón para confirmar la respuesta
if st.button("Validar relación", key="val_scatter"):
    # Aquí se puede personalizar la respuesta esperada dependiendo de la relación de las columnas seleccionadas
    st.markdown(f"Has seleccionado: {respuesta_scatter}. Observa el gráfico para evaluar si coincide con la relación entre `{columna_x}` y `{columna_y}`.")
    if respuesta_scatter == 'Relación positiva':
        st.success("¡Correcto! Es una relación positiva.")
    else:
        st.error("Incorrecto. Observa que a medida que aumenta el valor de 'PowerConsumption_Zone1' también aumenta el valor de 'PowerConsumption_Zone2'.")


# Sección: Diagrama de Pastel de "Temperature" discretizado
with st.container():
    st.header("8. Diagrama de Pastel de Temperatura")

    st.write("""
    El diagrama de pastel es una representación gráfica que muestra la proporción de distintas categorías dentro de un conjunto de datos.
    Cada porción del gráfico representa una categoría y su tamaño refleja la proporción que esta categoría ocupa respecto al total. 
    Es útil para visualizar cómo se distribuyen los datos en diferentes grupos o categorías.

    En este caso, no contamos con variables categóricas en su forma original, por lo que vamos a dividir los valores de la columna de temperatura en rangos de 10 grados. 
    Así podremos observar cómo se distribuyen los datos de temperatura en diferentes intervalos.
    """)

    # Discretizar la columna "Temperature" en rangos de 10 grados
    bins_temp = range(int(consumption['Temperature'].min()), int(consumption['Temperature'].max()) + 10, 10)

    # Generar las etiquetas para cada intervalo automáticamente
    labels_temp = pd.IntervalIndex.from_breaks(bins_temp).astype(str)

    # Aplicar pd.cut() para discretizar los valores de 'Temperature' en 'Temperature_Rango'
    consumption['Temperature_Rango'] = pd.cut(consumption['Temperature'], bins=bins_temp, labels=labels_temp, right=False)

    # Mostrar el diagrama de pastel para la temperatura discretizada
    fig_pie_temp, ax_pie_temp = plt.subplots()
    valores_temp = consumption['Temperature_Rango'].value_counts()
    ax_pie_temp.pie(valores_temp, labels=valores_temp.index, autopct='%1.1f%%', startangle=90)
    ax_pie_temp.axis('equal')  # Para asegurar que el pastel sea circular
    ax_pie_temp.set_title('Diagrama de Pastel de Temperatura Discretizada')

    # Mostrar la gráfica en la app
    st.pyplot(fig_pie_temp)

    st.markdown(f"""
    <div style="text-align: right;">
    <small> Salida generada por <code>plt.pie(valores_temp)</code></small>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta sobre el rango de menor representación
    st.markdown("### Pregunta:")
    st.markdown("Observa el diagrama de pastel. ¿Cuál de los siguientes rangos de temperatura tiene la menor representación?")

    # Crear opciones basadas en los rangos de temperatura
    opciones_temp = labels_temp

    # Crear el radio button para seleccionar el rango
    respuesta_temp = st.radio("Selecciona una opción:", opciones_temp)

    # Botón para confirmar la respuesta
    if st.button("Validar respuesta", key="validar_pastel"):
        if respuesta_temp == opciones_temp[-1]:  # La última opción es la correcta
            st.success(f"¡Correcto! El rango {respuesta_temp} tiene la menor representación en la temperatura según el diagrama de pastel.")
        else:
            st.error(f"Incorrecto. El rango con menor representación es el más pequeño en la gráfica de pastel.")

# Sección: Gráfico Boxplot con Plotly Express
with st.container():
    st.header("9. Gráfico Boxplot con `px.box`")

    # Descripción del boxplot
    st.markdown("""
        
    Un gráfico de caja o **boxplot** es una visualización gráfica que resume la distribución de un conjunto de datos numéricos a través de sus cuartiles. 
    Es útil para identificar valores atípicos, la dispersión y la simetría en la distribución de los datos.

    Los elementos principales del boxplot son:

    - **Caja**: Representa el rango intercuartílico (IQR), que va desde el primer cuartil (Q1) hasta el tercer cuartil (Q3). La línea dentro de la caja indica la mediana.
    - **Bigotes**: Extienden desde los cuartiles hacia los valores mínimos y máximos dentro de 1.5 veces el IQR.
    - **Puntos fuera de los bigotes**: Son considerados valores atípicos.

    El **boxplot** permite comparar la dispersión y los valores atípicos de diferentes columnas o grupos en los datos.
    """)

    # Seleccionar la columna para el boxplot
    columna_boxplot = st.selectbox('Selecciona la columna para el boxplot:', columnas_numericas)

    # Mostrar el boxplot
    fig_boxplot = px.box(consumption.iloc[np.random.randint(0,len(consumption), 5000)], y=columna_boxplot, points="all")
    fig_boxplot.update_layout(title=f'Boxplot de {columna_boxplot}', yaxis_title=columna_boxplot)

    # Mostrar la gráfica en la app
    st.plotly_chart(fig_boxplot)

    st.markdown(f"""
    <div style="text-align: right;">
    <small>Salida generada por <code>px.box(consumption, y='{columna_boxplot}')</code></small>
    </div>
    """, unsafe_allow_html=True)

# Pregunta interactiva sobre el boxplot
st.markdown("### Pregunta:")
st.markdown(f"Observa el boxplot de la columna 'DiffuseFlows'. ¿Qué puedes decir sobre los datos?")

# Opciones de análisis del boxplot
opciones_boxplot = ['Presencia de valores atípicos', 'Sin valores atípicos']

# Crear el radio button sin selección predeterminada
respuesta_boxplot = st.radio("Selecciona una opción:", opciones_boxplot)

# Botón para confirmar la respuesta
if st.button("Validar análisis", key="val_boxplot"):
    # Aquí puedes personalizar la respuesta esperada dependiendo de la columna seleccionada
    if respuesta_boxplot == 'Presencia de valores atípicos':
        st.success("¡Correcto! Se observan valores atípicos.")
    else:
        st.error("Revisa nuevamente los datos en el boxplot, quizás haya otros aspectos importantes que notar.")

# Sección: Gráfico de Violín con Plotly Express
with st.container():
    st.header("10. Gráfico de Violín con `px.violin`")

    # Descripción del diagrama de violín
    st.markdown("""
    
    Un **gráfico de violín** es una visualización que combina características del gráfico de caja (boxplot) y un gráfico de densidad. 
    Muestra la distribución de los datos numéricos y también su densidad (frecuencia) a lo largo de los valores. 
    Es útil para entender mejor la forma de la distribución, su asimetría, y la presencia de múltiples picos en los datos.

    Los elementos principales del diagrama de violín son:

    - **Cuerpo del violín**: Representa una estimación de la densidad de los datos, mostrando las frecuencias relativas.
    - **Mediana y cuartiles**: Al igual que en un boxplot, se pueden mostrar la mediana y los cuartiles dentro del gráfico.
    - **Distribución bimodal o multimodal**: A diferencia del boxplot, el gráfico de violín puede indicar si los datos tienen más de un pico o modo en la distribución.

    Este gráfico proporciona información adicional sobre la distribución de los datos, más allá de lo que muestra un boxplot.
    """)

    # Seleccionar la columna para el gráfico de violín
    columna_violin = st.selectbox('Selecciona la columna para el gráfico de violín:', columnas_numericas)

    # Mostrar el gráfico de violín
    fig_violin = px.violin(consumption.iloc[np.random.randint(0,len(consumption), 5000)], y=columna_violin, box=True, points="all")
    fig_violin.update_layout(title=f'Gráfico de Violín de {columna_violin}', yaxis_title=columna_violin)

    # Mostrar la gráfica en la app
    st.plotly_chart(fig_violin)

    st.markdown(f"""
    <div style="text-align: right;">
    <small>Salida generada por <code>px.violin(consumption, y='{columna_violin}', box=True, points="all")</code></small>
    </div>
    """, unsafe_allow_html=True)

# Pregunta interactiva sobre el gráfico de violín
st.markdown("### Pregunta:")
st.markdown("""Observa el gráfico de violín de la columna 'PowerConsumption_Zone3'. Teniendo en cuenta que una distribución 
simétrica significa que los datos están distribuidos de manera equilibrada alrededor de la mediana y la distribución asimétrica el caso contrario. ¿Qué puedes decir sobre los datos?""")

# Opciones de análisis del diagrama de violín
opciones_violin = ['Distribución simétrica', 'Distribución asimétrica']

# Crear el radio button sin selección predeterminada
respuesta_violin = st.radio("Selecciona una opción:", opciones_violin)

# Botón para confirmar la respuesta
if st.button("Validar análisis", key="val_violin"):
    # Aquí puedes personalizar la respuesta esperada dependiendo de la columna seleccionada
    if respuesta_violin == 'Distribución asimétrica':
        st.success("¡Correcto! La distribución es asimétrica.")
    else:
        st.error("Revisa nuevamente los datos en el gráfico de violín para identificar la forma de la distribución. bserva que no es igual arriba y abajo.")



# Mensaje de cierre del módulo
st.write("¡Fin del módulo 1! Ahora ya sabes cómo hacer una exploración gráfica de datasets.")

st.warning("Cuando cierres esta ventana, no podrás guardar tu progreso. Si quieres tener una copia de tu trabajo, dirígete a la esquina superior derecha, donde verás tres puntos. Selecciona la opción 'Print' para guardar tus avances en formato PDF.")