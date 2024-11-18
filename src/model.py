import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.etl import load_data  

# Función para entrenar y evaluar un modelo con las variables más importantes
def train_and_evaluate_model(data, target_column, important_features):
    # Separar variables independientes (X) y dependientes (y)
    X = data[important_features]
    y = data[target_column]
    
    # Codificar variables categóricas si es necesario
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # Entrenar un modelo Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return rf, accuracy

# Seleccionamos las 3 variables importantes con las que entrenamos el modelo
diabetes_features_imp = ['BMI', 'Age', 'GenHlth']
hypertension_features_imp = ['cp', 'thalach', 'oldpeak']

# Función para entrenar modelos con las características seleccionadas
def train_models():
    df_diabetes, df_hypertension = load_data()

    diabetes_model, diabetes_accuracy = train_and_evaluate_model(
        df_diabetes, 'Diabetes', diabetes_features_imp
    )

    hypertension_model, hypertension_accuracy = train_and_evaluate_model(
        df_hypertension, 'target', hypertension_features_imp
    )

    return {
        'diabetes': (diabetes_model, diabetes_accuracy),
        'hypertension': (hypertension_model, hypertension_accuracy)
    }
