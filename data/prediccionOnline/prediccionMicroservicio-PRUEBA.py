import pickle
from flask import Flask, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# Load the trained scikit-learn models stored in pickle format
with open('data/prediccionOnline/travelModel.pkl', 'rb') as f:
    modelo_tiempo_viaje = pickle.load(f)

with open('data/prediccionOnline/deliveryModel.pkl', 'rb') as f:
    modelo_tiempo_entrega = pickle.load(f)

with open('data/prediccionOnline/le.pkl', 'rb') as f:
    labelEncoder = pickle.load(f)



# Endpoint for route prediction model
# Input is a json object with attribute time
@app.route('/predict_eta', methods=['POST'])
def predict_eta():
    data = request.get_json()  # Obtener los datos JSON de la petición
    time_features = np.array(data['time']).reshape(1, -1)  # Asegurar la forma correcta para el modelo
    prediccion = modelo_tiempo_viaje.predict(time_features)  # Realizar la predicción
    return jsonify({'prediction': prediccion[0].tolist()})  # Devolver la predicción como respuesta JSON


# Endpoint for load delivery endpoint.
# Input is a json object with attributes truckId and time
@app.route('/predict_delivery', methods=['POST'])
def predict_delivery(): ## MODIFICADA !! para solo usar time o truckId dado que el modelo solo toma una feature
    
# Realizar la predicción utilizando solo truckId
    data = request.get_json()
    truck_id = np.array([data['truckId']])
    encoded_truck_id = labelEncoder.transform(truck_id).reshape(-1, 1)
    prediccion = modelo_tiempo_entrega.predict(encoded_truck_id)
    return jsonify({'prediction': prediccion[0].tolist()})

# # Realizar la predicción utilizando solo time
#     data = request.get_json()
#     time_features = np.array(data['time']).reshape(1, -1)  # Extraer y dar forma a las características de tiempo
#     prediccion = modelo_tiempo_entrega.predict(time_features) #solo cojo una feature, porque cuando intento tb con truckId me sale error en Postman (arreglar)
#     return jsonify({'prediction': prediccion[0].tolist()})  # Devolver la predicción como respuesta JSON

if __name__ == '__main__':
    app.run(debug=True, port =7777, host='0.0.0.0')