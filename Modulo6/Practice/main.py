
###################################
#### IMPORTAMOS LIBRERIAS #########
###################################
import warnings
warnings.filterwarnings("ignore")
# Manipulación de datos
import numpy as np
import pandas as pd
# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
# Configuración de gráficos
sns.set()
# Reducción de dimensionalidad
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
# Preprocesamiento de datos
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
# Modelos de selección y evaluación
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# Modelos de aprendizaje automático
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# Métricas de evaluación
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
# Guardar o cargar pesos del modelo 
import joblib
# PROYECTO BACKEND
from flask import Flask, request, jsonify

################################
# GENERAMOS LA APLICACIÓN ######
################################

app = Flask(__name__)


################################
# CARGAMOS EL MODELO ENTRENADO##
################################

MODEL = joblib.load('./weights/logistic_regression_model.pkl')


################################
# GENERAMOS LAS APIS DE ACCESO #
################################

@app.route('/getData', methods=['GET'])
def getData():
    """
    Endpoint para obtener los datos con los cuales testear un modelo
    """
    try:
        loaded_inputs = np.load('./datasets/test_data.npy')
        loaded_labels = np.load('./datasets/test_labels.npy')
        # Responder con los resultados
        return jsonify({
            "inputs": loaded_inputs.tolist(),
            "labels": loaded_labels.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/calculate_accuracy', methods=['GET'])
def calculateAccuracy():
     """
     Endpoint para calcular el accuracy actual del modelo con una base de datos de pruebas.
     Recibe datos en formato JSON y devuelve las predicciones.
     """
     loaded_inputs = np.load('./datasets/test_data.npy')
     loaded_labels = np.load('./datasets/test_labels.npy')
     try:
        # Hacer predicciones en el conjunto de prueba
        y_pred = MODEL.predict(loaded_inputs)
        # Calcular la precisión
        accuracy = accuracy_score(loaded_labels, y_pred)
        # Responder con los resultados
        return jsonify({
            "accuracy": accuracy
        })
     except Exception as e:
         return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run()





