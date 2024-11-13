from django.shortcuts import render
from django.conf import settings
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

def main(request):
    return render(request, 'index.html')

def prediccion(request):
    LOGISTIC_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'modelo_regresion_logistica.pkl')
    TREE_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'modelo_arbol_decision.pkl')
    
    data = pd.read_csv("/Users/ortiz/OneDrive/Escritorio/django_ml/proyecto/data/heart.csv", sep=";")
    
    # Estandarización de variables numéricas
    scaler = StandardScaler()
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Dividir en características (X) y variable objetivo (y)
    X = data.drop('target', axis=1)
    y = data['target']

    # Selección de características con SelectKBest
    selector = SelectKBest(score_func=f_classif, k=5)
    X_selected = selector.fit_transform(X, y)

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # Balanceo de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Ajuste de hiperparámetros para Regresión Logística
    param_grid_log = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
    grid_log = GridSearchCV(LogisticRegression(), param_grid_log, cv=5, scoring='f1')
    grid_log.fit(X_train_balanced, y_train_balanced)
    log_reg = grid_log.best_estimator_
    y_pred_log = log_reg.predict(X_test)
    joblib.dump(log_reg, LOGISTIC_MODEL_PATH)

    # Ajuste de hiperparámetros para Árbol de Decisión
    param_grid_tree = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5, scoring='f1')
    grid_tree.fit(X_train_balanced, y_train_balanced)
    tree_clf = grid_tree.best_estimator_
    y_pred_tree = tree_clf.predict(X_test)
    joblib.dump(tree_clf, TREE_MODEL_PATH)

    # Evaluación de la Regresión Logística
    log_reg_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_log),
        "recall": recall_score(y_test, y_pred_log),
        "f1_score": f1_score(y_test, y_pred_log),
        "roc_auc": roc_auc_score(y_test, y_pred_log),
        "confusion_matrix": confusion_matrix(y_test, y_pred_log).tolist(),
    }

    # Evaluación del Árbol de Decisión
    tree_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_tree),
        "recall": recall_score(y_test, y_pred_tree),
        "f1_score": f1_score(y_test, y_pred_tree),
        "roc_auc": roc_auc_score(y_test, y_pred_tree),
        "confusion_matrix": confusion_matrix(y_test, y_pred_tree).tolist(),
    }

    # Datos para gráficas
    test_values = y_test.tolist()
    log_reg_predictions = y_pred_log.tolist()
    tree_predictions = y_pred_tree.tolist()

    context = {
        "log_reg_metrics": log_reg_metrics,
        "tree_metrics": tree_metrics,
        "test_values": test_values,
        "log_reg_predictions": log_reg_predictions,
        "tree_predictions": tree_predictions,
    }

    return render(request, 'index.html', context)
