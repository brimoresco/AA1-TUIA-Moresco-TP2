import pickle
import pandas as pd
import numpy as np

# Cargar el modelo, el scaler y el imputador (si los usas)
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
imputer = pickle.load(open('model/imputer.pkl', 'rb'))

def preprocesar_datos(datos_entrada):
    # Imputar valores faltantes si es necesario
    datos_entrada_imputados = imputer.transform(datos_entrada)
    
    # Escalar los datos de entrada
    datos_entrada_scaled = scaler.transform(datos_entrada_imputados)
    
    return datos_entrada_scaled

def realizar_prediccion(datos_entrada):
    datos_preprocesados = preprocesar_datos(datos_entrada)
    prediccion = model.predict(datos_preprocesados)
    return prediccion

# Simulando entrada de datos (esto sería reemplazado por la entrada de la API o un archivo)
if __name__ == "__main__":
    # Supón que tienes un dataframe de datos de entrada (X_new)
    datos_entrada = pd.DataFrame([[5.2, 3.4, 1.5, 0.2]], columns=["feature1", "feature2", "feature3", "feature4"])
    predicciones = realizar_prediccion(datos_entrada)
    print(f"Predicción: {predicciones}")
