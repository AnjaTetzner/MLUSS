import local_regression
import local_classifier
import local_plotting
import local_dataManagement
import randomizedSearch


from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from tabulate import tabulate   

import csv

# multithreading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


import numpy as np

def repeat_training_and_evaluation(data,
                                    repetitions=100,
                                    model=MLPRegressor(hidden_layer_sizes=(100), max_iter=10000)):
    """
    Führt das Training und die Bewertung von Regressionsmodellen mehrfach durch und berechnet 
    Statistikmetriken (Mean, Median, Standardabweichung) für die Ergebnisse.

    Parameter:
        data (DataFrame): Der Datensatz, der für die Vorbereitung und das Training verwendet wird.
        repetitions (int): Anzahl der Wiederholungen (Standard: 100).

    Rückgabe:
        results (dict): Statistikmetriken (mean, median, std) für alle Modelle.
    """
    metrics_rmse = {
        "Gutschweißung": [],
        "Leitungsversatz": [],
        "Sontrodeneinsätze": [],
        "Terminalversatz": [],
        "Öl auf Terminal": []
    }
    
    metrics_rSquare = {
        "Gutschweißung": [],
        "Leitungsversatz": [],
        "Sontrodeneinsätze": [],
        "Terminalversatz": [],
        "Öl auf Terminal": []
    }
    
    for i in range(repetitions):
        print(f"\rIteration {i + 1}/{repetitions}", end="")
        
        # Daten vorbereiten
        X_01, y_01 = local_regression.prepare_data(data, "Gutschweißung")
        X_02, y_02 = local_regression.prepare_data(data, "Leitungsversatz")
        X_03, y_03 = local_regression.prepare_data(data, "Sontrodeneinsätze")
        X_04, y_04 = local_regression.prepare_data(data, "Terminalversatz")
        X_05, y_05 = local_regression.prepare_data(data, "Öl auf Terminal")

        # Modell trainieren und Metriken speichern
        _, metrics01 = local_regression.train_model(X_01, y_01, model=model)
        _, metrics02 = local_regression.train_model(X_02, y_02, model=model)
        _, metrics03 = local_regression.train_model(X_03, y_03, model=model)
        _, metrics04 = local_regression.train_model(X_04, y_04, model=model)
        _, metrics05 = local_regression.train_model(X_05, y_05, model=model)
        
        metrics_rmse["Gutschweißung"].append(metrics01['RMSE'])
        metrics_rmse["Leitungsversatz"].append(metrics02['RMSE'])
        metrics_rmse["Sontrodeneinsätze"].append(metrics03['RMSE'])
        metrics_rmse["Terminalversatz"].append(metrics04['RMSE'])
        metrics_rmse["Öl auf Terminal"].append(metrics05['RMSE'])
    
        metrics_rSquare["Gutschweißung"].append(metrics01['R2_Score'])
        metrics_rSquare["Leitungsversatz"].append(metrics02['R2_Score'])
        metrics_rSquare["Sontrodeneinsätze"].append(metrics03['R2_Score'])
        metrics_rSquare["Terminalversatz"].append(metrics04['R2_Score'])
        metrics_rSquare["Öl auf Terminal"].append(metrics05['R2_Score'])

    # Statistiken berechnen
    results_rmse = {}
    for category, metric_values in metrics_rmse.items():
        results_rmse[category] = {
            "mean": np.mean(metric_values),
            "median": np.median(metric_values),
            "std": np.std(metric_values)
        }
    
    results_rSquared = {}
    for category, metric_values in metrics_rSquare.items():
        results_rSquared[category] = {
            "mean": np.mean(metric_values),
            "median": np.median(metric_values),
            "std": np.std(metric_values)
        }
    
    return results_rmse, results_rSquared

# Verwendung der Klasse
if __name__ == "__main__":
    #file_path = "complete_data.csv"
    file_path ="../daten/normalized_complete_data.csv"
    file_path ="../daten/corrected_normalized_P_F_D_complete_data.csv"
    file_path ="../daten/corrected_normalized_P_F_D_complete_data_noZeroZ.csv"
    
    data_storage = local_dataManagement.DataStorage(file_path)

    # Daten laden
    data_storage.load_data()

    # Geladene Daten anzeigen
    data = data_storage.get_data()

    #-------------------
    # classification

    ## Daten vorbereiten
    #X, y = local_classifier.prepare_classification_data(data)
    
    # Klassifikationsmodell trainieren
    #rf_model = local_classifier.classify_data(X, y)

    #-------------------
    # regression
    X, y = local_regression.prepare_data(data)

   # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "mlp": MLPRegressor(hidden_layer_sizes=(100), max_iter=5000),
 #       'svr01': SVR(),
    }

    param_distributions = {
        "mlp": {"alpha": np.logspace(-4, 0, 50)},
  #      'svr01': {
   #             'C': [0.1, 1, 10, 100],
   #             'epsilon': [0.1, 0.2, 0.5],
   #             'kernel': ['linear', 'rbf', 'poly']
   #                 }
    }
#        'svr01': {
#                  'C': np.logspace(-3, 3, 10),            # Regularisierung: von sehr klein bis sehr groß
#                  'epsilon': np.logspace(-4, 1, 10),      # Epsilon-Toleranz
#                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Verschiedene Kernel-Typen
#                  'degree': [2, 3, 4, 5],  # Grad für den 'poly'-Kernel
#                  'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 10))}} # Gamma-Wert (für RBF, poly, sigmoid)
#    

    #model, metrics = local_regression.train_model_normalize(X_train, y_train,model = MLPRegressor(hidden_layer_sizes=(100), max_iter=5000, alpha=0.0024420530945486497))


#    model, metrics = local_regression.train_model(X_train, y_train,model = MLPRegressor(hidden_layer_sizes=(100), max_iter=5000, alpha=0.0024420530945486497))
    #print(metrics)

    #Assuming X_train and y_train are defined
    #results = randomizedSearch.randomized_search_cv_with_logging(models, X_train, y_train, param_distributions,
    #          scoring='neg_mean_absolute_error')

    # Normalized X, normal Y
#    results = randomizedSearch.randomized_search_xTrainNormalized(models, X_train, y_train, param_distributions,
#               scoring='neg_mean_absolute_error')

    # Normalized X, normal y; print mae for each fold
    results = randomizedSearch.randomized_search_xTrainNormalized_mae(models, X_train, y_train, param_distributions,
              scoring='neg_mean_absolute_error')
    
    print(results)
