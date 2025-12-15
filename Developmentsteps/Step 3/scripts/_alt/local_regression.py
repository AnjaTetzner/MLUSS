import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

#def prepare_data(data, 
#                classification):

def prepare_data(data):
    """
    Aggregiert die Prozessdaten für das Training eines Modells.
    
    Parameter:
        data (DataFrame): Der vollständige Datensatz mit P, F, D und Experiment_ID.
    
    Rückgabe:
        X (DataFrame): Aggregierte Eingabe-Features.
        y (Series): Zielvariable Z.
    """
    # Ergebnisse speichern
    features = []
    targets = []
    
    # Gruppieren nach Experiment_ID
    for experiment_id, group in data.groupby('Experiment_ID'):

         # Filtern: Nur Reihen mit "Gut" in der Spalte "Kategorie"
         # Leitungsversatz
         # Öl auf Terminal
         # Sontrodeneins
        #classification="Gutschweißung"
        #filtered_group = group[group['Kategorie'].str.contains(classification, na=False)]
        
        # Überspringen, wenn nach dem Filtern keine Daten übrig sind
        #if filtered_group.empty:
          #  print(f"Experiment_ID {experiment_id} übersprungen (keine 'Gut'-Reihen vorhanden).")
        #    continue

        # Zielvariable Z aus der letzten Zeile extrahieren
        target = group.iloc[-1]['Z']
        targets.append(target)
        
        # Aggregation der Eingabefeatures
        feature_row = {
            'P_mean': group['P'].mean(),
            'P_sum': group['P'].min(),
            'P_max': group['P'].max(),
            'F_mean': group['F'].mean(),
            'F_sum': group['F'].min(),
            'F_max': group['F'].max(),
            'D_mean': group['D'].mean(),
            'D_sum': group['D'].min(),
            'D_max': group['D'].max(),
        }

        features.append(feature_row)
    
    # Features und Zielvariable in DataFrames umwandeln
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Z')
    
    return X, y
    

def train_model(X,y,model = MLPRegressor(hidden_layer_sizes=(100), max_iter=10000)):
    """
    Erstellt und trainiert ein Regressionsmodell auf Basis der Spalten P, F, und D, um Z vorherzusagen.

    Parameter:
        data (DataFrame): Der Datensatz mit den Spalten P, F, D und Z.

    Rückgabe:
        model (LinearRegression): Das trainierte Regressionsmodell.
        metrics (dict): Metriken zur Bewertung des Modells auf dem Testdatensatz.
    """

    # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #model = MLPRegressor(hidden_layer_sizes=(100), max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    metrics = {
        'R2_Score': r2_score(y_test, y_pred),
        'MSE': mse,
        'MAE': mae
    }

    return model, metrics

def train_model_normalize(X, y_original, model=MLPRegressor(hidden_layer_sizes=(100), max_iter=10000)):
    """
    Erstellt und trainiert ein Regressionsmodell auf Basis der Spalten P, F, und D, um Z vorherzusagen.

    Parameter:
        X (DataFrame): Feature-Matrix mit den Spalten P, F, D.
        y_original (Series): Zielvariable (Z) in unnormalisierter Form.
        model (Regressor): Das zu trainierende Modell (Standard: MLPRegressor).

    Rückgabe:
        model (Regressor): Das trainierte Regressionsmodell.
        metrics (dict): Metriken zur Bewertung des Modells auf dem Testdatensatz.
    """

    # Normalisierung der Zielvariable
    y_mean = np.mean(y_original)
    y_std = np.std(y_original)
    y_normalized = (y_original - y_mean) / y_std

    # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2)

    # Modell trainieren
    model.fit(X_train, y_train)

    # Vorhersagen auf Testdaten
    y_pred_normalized = model.predict(X_test)

    # Rückskalierung der Vorhersagen
    y_pred = (y_pred_normalized * y_std) + y_mean

    # Metriken berechnen
    mse = mean_squared_error((y_test * y_std) + y_mean, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error((y_test * y_std) + y_mean, y_pred)
    metrics = {
        'R2_Score': r2_score((y_test * y_std) + y_mean, y_pred),
        'MSE': mse,
        'MAE': mae
    }

    return model, metrics
