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
                          'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
                          'Cloud3pm', 'Temp9am', 'Temp3pm']

def agrupar_direcciones(direccion):
    """
    Agrupa las direcciones en 'N', 'S', 'E', 'W', o 'Otro'.
    """
    for grupo, direcciones in diccionario.items():
        if direccion in direcciones:
            return grupo
    return "Otro"


def preprocesar(data):
    print("Columnas esperadas por el modelo:", model.feature_names_in_)
    print("Columnas presentes en los datos iniciales:", data.columns)

    # 1. Normalizar formato numérico
    for col in data.select_dtypes(include=['object']).columns:
        try:
            data[col] = data[col].str.replace('.', '', regex=True).str.replace(',', '.', regex=True).astype(float)
        except ValueError:
            pass  # Ignorar columnas que no sean numéricas

    # 2. Mapear direcciones de viento
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if col in data.columns:
            data[col] = data[col].map(diccionario_invertido).fillna(data[col])

    # 3. Imputación de valores faltantes
    numeric_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(include='object').columns

    # Imputar columnas numéricas
    data[numeric_cols] = data[numeric_cols].fillna(median_values)
    if numeric_cols:
        data[numeric_cols] = knn_imputer.transform(data[numeric_cols])

    # Imputar columnas categóricas
    data[categorical_cols] = data[categorical_cols].fillna(mode_values)

    # 4. Aplicar One Hot Encoding
    columns_to_dummy = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    columns_to_dummy = [col for col in columns_to_dummy if col in data.columns]
    if columns_to_dummy:
        data = pd.get_dummies(data, columns=columns_to_dummy, drop_first=True)

    # 5. Validar consistencia con el modelo
    columnas_entrenamiento = model.feature_names_in_
    missing_cols = set(columnas_entrenamiento) - set(data.columns)
    unexpected_cols = set(data.columns) - set(columnas_entrenamiento)

    # Agregar columnas faltantes con valores predeterminados
    for col in missing_cols:
        data[col] = 0

    # Eliminar columnas inesperadas
    data = data.drop(columns=unexpected_cols, errors='ignore')

    # Ordenar las columnas para que coincidan con el modelo
    data = data[columnas_entrenamiento]

    # Guardar log detallado
    with open('columns_debug_log.txt', 'w') as f:
        f.write("Columnas presentes en el DataFrame preprocesado:\n")
        f.write('\n'.join(data.columns.tolist()))
        f.write("\n\nColumnas faltantes respecto al modelo:\n")
        f.write('\n'.join(list(missing_cols)))
        f.write("\n\nColumnas inesperadas en los datos:\n")
        f.write('\n'.join(list(unexpected_cols)))

    print("Las columnas se han guardado en columns_debug_log.txt")

    # 6. Estandarizar columnas
    if columns_to_standardize:
        data[columns_to_standardize] = scaler.transform(data[columns_to_standardize])
    
    return data



# Predicción
def predecir(data):
    data_preprocesada = preprocesar(data.copy())
    return model.predict(data_preprocesada)


# Punto de entrada
if __name__ == '__main__':
    import pandas as pd

    print("Ingrese la ruta del archivo CSV:")
    input_file = input("> ")

    try:
        # Leer el archivo CSV
        data = pd.read_csv(input_file)

        # Asegurar que todas las columnas requeridas estén presentes
        missing_cols = set(model.feature_names_in_) - set(data.columns)
        for col in missing_cols:
            data[col] = 0

        # Predecir y guardar resultados
        predictions = predecir(data)
        data['Prediction'] = predictions
        output_file = 'output.csv'
        data.to_csv(output_file, index=False)
        print(f"Predicciones guardadas en {output_file}")

    except FileNotFoundError:
        print("Error: El archivo especificado no fue encontrado.")
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío o tiene un formato incorrecto.")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")


