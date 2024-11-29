import joblib
import pandas as pd
import sys

# Cargar los objetos serializados
model = joblib.load('docker/model/modelOp.joblib')
scaler = joblib.load('docker/model/scaler.joblib')
knn_imputer = joblib.load('docker/model/knn_imputer.joblib')
median_values = joblib.load('docker/model/median_values.joblib')
mode_values = joblib.load('docker/model/mode_values.joblib')


# Diccionario de transformación para direcciones de viento
diccionario = {
    'N': ['N', 'NNW', 'NNE', 'NE', 'NW'],
    'S': ['S', 'SSW', 'SSE', 'SE', 'SW'],
    'E': ['E', 'ENE', 'ESE'],
    'W': ['W', 'WNW', 'WSW'],
}
diccionario_invertido = {valor: clave for clave, lista_valores in diccionario.items() for valor in lista_valores}

# Columnas a estandarizar
columns_to_standardize = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
       'Temp9am', 'Temp3pm']


# Preprocesamiento
def preprocesar(data):
    # 1. Mapear direcciones de viento
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if col in data.columns:
            data[col] = data[col].map(diccionario_invertido).fillna(data[col])
    
    
    # 2. Imputación de valores faltantes
    numeric_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(include='object').columns
    
    # Validar columnas conocidas por el imputador
    numeric_cols = [col for col in numeric_cols if col in knn_imputer.feature_names_in_]
    data[numeric_cols] = data[numeric_cols].fillna(median_values)
    data[numeric_cols] = knn_imputer.transform(data[numeric_cols])
    data[categorical_cols] = data[categorical_cols].fillna(mode_values)
    
    # 3. Aplicar One Hot Encoding
    columns_to_dummy = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    columns_to_dummy = [col for col in columns_to_dummy if col in data.columns]
    if columns_to_dummy:
        data = pd.get_dummies(data, columns=columns_to_dummy, drop_first=True)
    
    # 4. Asegurar consistencia con las columnas del modelo
    columnas_entrenamiento = scaler.feature_names_in_
    missing_cols = set(columnas_entrenamiento) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    data = data[columnas_entrenamiento]  
    # Después de la función `preprocesar` o dentro de ella después de `data = reindex(...)`
    missing_cols = set(columnas_entrenamiento) - set(data.columns)

    # Guardar el log de columnas en un archivo
    with open('columns_log.txt', 'w') as f:
        f.write("Columnas presentes en el DataFrame preprocesado:\n")
        f.write('\n'.join(data.columns.tolist()))
        f.write("\n\nColumnas faltantes respecto al modelo:\n")
        f.write('\n'.join(list(missing_cols)))
    print("Las columnas se han guardado en columns_log.txt")

   
    # 5. Estandarizar columnas
    data[columns_to_standardize] = scaler.transform(data[columns_to_standardize])
    return data



# Predicción
def predecir(data):
    data_preprocesada = preprocesar(data)
    return model.predict(data_preprocesada)

# Punto de entrada
if __name__ == '__main__':
    print("Seleccione una opción:")
    print("1. Predecir desde archivo CSV")
    print("2. Ingresar datos manualmente")

    opcion = input("Ingrese el número de opción: ")

    if opcion == '1':
        input_file = input("Ingrese la ruta del archivo CSV: ")
        data = pd.read_csv(input_file)
        predictions = predecir(data)
        data['Prediction'] = predictions
        output_file = 'output.csv'
        data.to_csv(output_file, index=False)
        print(f"Predicciones guardadas en {output_file}")

    elif opcion == '2':
        user_data = {}
        for col in scaler.feature_names_in_:
            try:
                val = input(f"{col}: ")
                user_data[col] = float(val) if val.replace('.', '', 1).isdigit() else val
            except ValueError:
                user_data[col] = None
        
        data = pd.DataFrame([user_data])
        predictions = predecir(data)
        print(f"Predicción: {'Lloverá' if predictions[0] == 1 else 'No lloverá'}")

    else:
        print("Opción no válida. Por favor, intente de nuevo.")
