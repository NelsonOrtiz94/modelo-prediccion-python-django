# Se importan las librerias para el template y los renders
from django.shortcuts import render
import json


# Librerias para operaciones matemáticas
import numpy as np
# Libreria para el manejo de datos
import pandas as pd


# Libreria para cambio de datos
from sklearn.preprocessing import LabelEncoder
# Libreria para balanceo de los datos
from sklearn.utils import resample
# Libreria para separar los datos de entrenamiento y pruebas
from sklearn.model_selection import train_test_split
# Libreria para la predicción con árbol de decisión 
from sklearn.tree import DecisionTreeClassifier
# Libreria para las métricas
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------------

def main(request):
    # *** Plantilla ***
    return render(request, 'index.html', context={})



def prediccion(request):

    ### Se cargan los datos
    data = pd.read_csv("/home/jose/Documentos/GitHub/django_ml/proyecto/data/jugar_clima_temperatura.csv", sep=";")

    ### Se modifica luvioso por lluvioso
    data = data.replace("luvioso", "lluvioso")

    ### Se cambian las cadenas a valores numéricos
    le = LabelEncoder()

    data['clima'] = le.fit_transform( data['clima'] )
    data['temperatura'] = le.fit_transform( data['temperatura'] )
    data['jugar'] = le.fit_transform( data['jugar'] )

    ### Se realiza el balanceo de los datos (oversample)
    df_cero = data[ data['jugar'] == 0 ]
    df_uno = data[ data['jugar'] == 1 ]

    df_oversample_cero = resample(df_cero,
                           replace=True,
                           n_samples=50,
                           random_state = 1)

    df_oversample_uno = resample(df_uno,
                           replace=True,
                           n_samples=50,
                           random_state = 1)

    df = pd.concat( [df_oversample_cero, df_oversample_uno] )


    ### Se definen las características y variable objetivo
    features = ['clima','temperatura']
    # Características
    X = df[features]
    # Variable objetivo
    y = df['jugar'].values

    # Se separan los datos (80% entrenamiento y 20% pruebas)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

    ### Se genera la predicción sin hiperparámetros
    # Se selecciona el algoritmo
    dt = DecisionTreeClassifier()

    # Se entrena el algoritmo
    dt.fit(X_train, y_train)

    # Se genera la predicción
    predict = dt.predict(X_test)

    # Se convierten los valores a listas
    test= y_test.tolist()
    prediction = predict.tolist()

    # Se envian los valores al contexto para renderizar
    context = {"test": test,
               "prediction": prediction}

    # *** Plantilla ***
    return render(request, 'index.html', context=context)
