from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


app = Flask(__name__)
# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo de clasificación
model = RandomForestClassifier(random_state=20)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para hacer predicciones.
    Recibe datos en formato JSON y devuelve las predicciones.
    """
    data = request.get_json()
    
    try:
        # Convertir datos de entrada en un DataFrame
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Realizar predicción
        prediction = model.predict(input_data)
        prediction_label = target_names[prediction[0]]
        
        # Responder con los resultados
        return jsonify({
            "prediction": int(prediction[0]),
            "label": prediction_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/test-data', methods=['GET'])
def test_data():
    """
    Endpoint para obtener un ejemplo de datos de prueba.
    """
    example = {feature_names[i]: X_test[0][i] for i in range(len(feature_names))}
    return jsonify(example)

if __name__ == '__main__':
    app.run()