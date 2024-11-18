import pandas as pd

# Función para cargar los datos
def load_data():
    df_diabetes = pd.read_csv('src/data/diabetes_data.csv')
    df_hypertension = pd.read_csv('src/data/hypertension_data.csv')

    # Llenar valores faltantes
    df_diabetes = df_diabetes.fillna(df_diabetes.median())
    df_hypertension = df_hypertension.fillna(df_hypertension.median())

    return df_diabetes, df_hypertension

# Crear un DataFrame con los nombres de las columnas
def prepare_patient_data_with_names(patient_data, feature_names):
    return pd.DataFrame([patient_data], columns=feature_names)

def categorizar_edad(edad):
    # Diccionario que mapea la categoría con el rango de edad
    categorias = {
        1: '0-24',
        2: '25-29',
        3: '30-34',
        4: '35-39',
        5: '40-44',
        6: '45-49',
        7: '50-54',
        8: '55-59',
        9: '60-64',
        10: '65-69',
        11: '70-74',
        12: '75-79',
        13: '80 o más'
    }
    
    # Comparar la edad del paciente y devolver la categoría adecuada
    if 0 <= edad <= 24:
        return 1
    elif 25 <= edad <= 29:
        return 2
    elif 30 <= edad <= 34:
        return 3
    elif 35 <= edad <= 39:
        return 4
    elif 40 <= edad <= 44:
        return 5
    elif 45 <= edad <= 49:
        return 6
    elif 50 <= edad <= 54:
        return 7
    elif 55 <= edad <= 59:
        return 8
    elif 60 <= edad <= 64:
        return 9
    elif 65 <= edad <= 69:
        return 10
    elif 70 <= edad <= 74:
        return 11
    elif 75 <= edad <= 79:
        return 12
    elif edad >= 80:
        return 13
    else:
        return None  # Una edad fuera de los rangos esperados
    