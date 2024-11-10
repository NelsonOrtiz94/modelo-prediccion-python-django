from django.shortcuts import render
from django.conf import settings
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def main(request):
    # Render de la página principal sin datos iniciales
    return render(request, 'index.html')

def prediccion(request):
    # Ruta de los modelos guardados
    LOGISTIC_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'modelo_regresion_logistica.pkl')
    TREE_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'modelo_arbol_decision.pkl')
    
    # Cargar los datos del archivo CSV
    data = pd.read_csv("/Users/ortiz/OneDrive/Escritorio/django_ml/proyecto/data/heart.csv", sep=";")
    
    # Estandarización de variables numéricas
    scaler = StandardScaler()
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Dividir en características (X) y variable objetivo (y)
    X = data.drop('target', axis=1)
    y = data['target']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Balanceo de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Modelo 1: Regresión Logística
    log_reg = LogisticRegression()
    log_reg.fit(X_train_balanced, y_train_balanced)
    y_pred_log = log_reg.predict(X_test)
    joblib.dump(log_reg, LOGISTIC_MODEL_PATH)  # Guardar el modelo

    # Modelo 2: Árbol de Decisión
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train_balanced, y_train_balanced)
    y_pred_tree = tree_clf.predict(X_test)
    joblib.dump(tree_clf, TREE_MODEL_PATH)  # Guardar el modelo

    # Evaluación del modelo de Regresión Logística
    log_reg_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_log),
        "recall": recall_score(y_test, y_pred_log),
        "f1_score": f1_score(y_test, y_pred_log),
        "roc_auc": roc_auc_score(y_test, y_pred_log),
        "confusion_matrix": confusion_matrix(y_test, y_pred_log).tolist(),
    }

    # Evaluación del modelo de Árbol de Decisión
    tree_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_tree),
        "recall": recall_score(y_test, y_pred_tree),
        "f1_score": f1_score(y_test, y_pred_tree),
        "roc_auc": roc_auc_score(y_test, y_pred_tree),
        "confusion_matrix": confusion_matrix(y_test, y_pred_tree).tolist(),
    }

    # Preparar datos para la gráfica de Highcharts
    test_values = y_test.tolist()
    log_reg_predictions = y_pred_log.tolist()
    tree_predictions = y_pred_tree.tolist()

    # Contexto para renderizar en la plantilla HTML
    context = {
        "log_reg_metrics": log_reg_metrics,
        "tree_metrics": tree_metrics,
        "test_values": test_values,
        "log_reg_predictions": log_reg_predictions,
        "tree_predictions": tree_predictions,
    }

    return render(request, 'index.html', context)
