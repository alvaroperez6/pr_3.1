import pandas as pd
import os



events=pd.read_json("../../simulation.jsonlines")
planes=pd.read_json("../../plans.jsonlines")

pd.DataFrame(planes["trucks"][0])

entregas = events[events.eventType.isin(["Truck ended delivering", "Truck started delivering"])]
entregas = entregas.sort_values(by=["simulationId", "truckId", "eventTime"])

entregas["deliverTime"] = entregas["eventTime"].diff()
entregas = entregas[entregas.eventType == "Truck ended delivering"]
entregas

events["eventType"].value_counts()

camiones_total = [  ]
for t in planes["trucks"]:
    camiones_total.append(pd.DataFrame(t))
camiones_total = pd.concat(camiones_total)
camiones_total["items"] = camiones_total["items"].apply(len)

camiones_total.drop(columns=["route"]).groupby("truck_id").sum()

localizaciones_total = [  ]
for truck_list in planes["trucks"]:
    for truck in truck_list:
        localizaciones_total.append(pd.DataFrame(truck["route"]).assign(truck_id=truck["truck_id"]))
    
localizaciones_total = pd.concat(localizaciones_total)
localizaciones_total
localizaciones_total[["truck_id","destination"]].groupby("truck_id").nunique()

tiempos_por_camion = [  ]
for sim_id in planes.simulationId.unique():
    for truck in planes[planes.simulationId == sim_id]["trucks"].values[0]:
        tiempos_por_camion.append(pd.DataFrame(truck["route"]).assign(truck_id=truck["truck_id"],simulationId=sim_id))
    
tiempos_por_camion = pd.concat(tiempos_por_camion)
tiempos_por_camion.drop(columns=["destination","origin"]).groupby(["truck_id","simulationId"]).agg(list)

events

events.eventType.unique()

tiempos_viaje = events[events.eventType.isin(["Truck departed", "Truck arrived", 'Truck departed to depot', 'Truck ended route'])]
tiempos_viaje = tiempos_viaje.drop(columns=["eventDescription"]).sort_values(["simulationId", "truckId","eventTime"])
tiempos_viaje["tiempo"] = tiempos_viaje["eventTime"].diff()
tiempos_viaje = tiempos_viaje[tiempos_viaje["eventType"].isin(["Truck arrived",'Truck ended route'])]
tiempos_viaje.drop(columns=["eventTime", "eventType"]).groupby(["truckId","simulationId"]).agg(list)

# Suponiendo que ya tienes los dataframes 'planes' y 'events'

# Paso 1: Obtener los tiempos de viaje de los camiones
# Se ha realizado una manipulación de los datos en el código anterior para obtener los tiempos de viaje, podemos continuar desde allí
tiempos_viaje_camiones = tiempos_viaje.drop(columns=["eventTime", "eventType"]).groupby(["truckId", "simulationId"]).agg(list)

# Paso 2: Obtener los tiempos de entrega de los paquetes
# Ya tienes un dataframe llamado 'entregas' que contiene los tiempos de entrega de los paquetes
# Puedes proceder a hacer cualquier manipulación adicional si es necesaria, por ejemplo, ajustar el formato de los datos, etc.

# Guardar los resultados en archivos CSV si es necesario
tiempos_viaje_camiones.to_csv("tiempos_viaje_camiones.csv")
entregas.to_csv("tiempos_entrega_paquetes.csv")
entregas

# Suponiendo que el dataframe con los planes se llama planes
df_planes_exploded = planes.join(planes.trucks.explode().apply(pd.Series), lsuffix='_sim').reset_index(drop=True)

# Ahora deberíamos repetir ese proceso con la columna 'route'
df_planes_exploded = df_planes_exploded.join(df_planes_exploded.route.explode().apply(pd.Series), lsuffix='_truck').reset_index(drop=True)

df_planes_exploded

# Suponiendo que tienes los dataframes planes y events
# Combinar los dataframes usando la función join
df_combinado = planes.join(events.set_index('simulationId'), on='simulationId', lsuffix='_planes', rsuffix='_eventos')

df_combinado

num_simulaciones = df_combinado['simulationId'].nunique()
num_eventos_total = df_combinado.shape[0]
num_planes_ejecutados = planes.shape[0]


eventos_por_tipo = df_combinado['eventType'].value_counts()

tiempos_viaje_camiones

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Paso 1: Obtener los tiempos de viaje de los camiones
tiempos_viaje_camiones = tiempos_viaje.drop(columns=["eventTime", "eventType"]).groupby(["truckId", "simulationId"]).agg(list)

# Desagregar la lista en la columna "duration"
tiempos_viaje_camiones_exploded = tiempos_viaje_camiones.explode("tiempo")

# Convertir la columna "duration" a un arreglo numpy y darle la forma adecuada
X_camiones = tiempos_viaje_camiones_exploded["tiempo"].values.reshape(-1, 1)

# Obtener las etiquetas (y) para el modelo
y_camiones = tiempos_viaje_camiones_exploded["tiempo"].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_camiones, X_test_camiones, y_train_camiones, y_test_camiones = train_test_split(X_camiones, y_camiones, test_size=0.2, random_state=42)

# Paso 3: Entrenar modelos de ML

# Entrenar modelo de regresión lineal para predecir el tiempo de viaje de los camiones
modelo_tiempo_viaje_camiones = LinearRegression()
modelo_tiempo_viaje_camiones.fit(X_train_camiones, y_train_camiones)

# Evaluar modelo de regresión lineal
y_pred_camiones = modelo_tiempo_viaje_camiones.predict(X_test_camiones)
mse_camiones = mean_squared_error(y_test_camiones, y_pred_camiones)
print("Error cuadrático medio (MSE) del modelo de tiempo de viaje de los camiones:", mse_camiones)

# Guardar modelo de regresión lineal para predecir el tiempo de viaje de los camiones
with open('modelo_tiempo_viaje_camiones.pkl', 'wb') as f:
    pickle.dump(modelo_tiempo_viaje_camiones, f)