import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import evaluation

# Modelle und weitere Imports
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tabulate import tabulate
import csv
import numpy as np
import pandas as pd
from datetime import datetime


# Schritt 1: Aggregation der Daten (mit erweiterter Feature-Berechnung für die PACs)
def prepare_data(data):
    """
    Aggregiert die Prozessdaten je Experiment und berechnet die benötigten Features
    für die Bildung der PACs (Principal Aggregate Components).

    Zusätzlich zu den bereits vorhandenen Features werden folgende aggregierte Größen berechnet:
      - D_mean: Mittelwert von D
      - D_IQR: Interquartilsabstand von D (0.75-Quantil minus 0.25-Quantil)
      - D_25: 25%-Quantil von D
      - D_75: 75%-Quantil von D

    Parameter:
        data (DataFrame): Der vollständige Datensatz mit Spalten wie P, F, D, Z und Experiment_ID.
    Rückgabe:
        X (DataFrame): Aggregierte Eingabe-Features.
        y (Series): Zielvariable Z.
    """
    features = []
    targets = []
    
    for experiment_id, group in data.groupby('Experiment_ID'):
        # Zielvariable: letzter Wert der Spalte Z
        target = group.iloc[-1]['Z']
        targets.append(target)

        feature_row = {
            # Power-Features
            'P_mean': group['P'].mean(),
            'P_max': group['P'].max(),
            'P_IQR': group['P'].quantile(0.75) - group['P'].quantile(0.25),
            
            # Force-Features
            'F_mean': group['F'].mean(),
            'F_max': group['F'].max(),
            'F_75': group['F'].quantile(0.75),
            
            # Distance-Features
            'D_mean': group['D'].mean(),         # zusätzlich benötigt für PAC3
            'D_std': group['D'].std(),
            'D_50': group['D'].median(),
            
            # Dynamic-Features
            'D_IQR': group['D'].quantile(0.75) - group['D'].quantile(0.25),  # zusätzlich benötigt für PAC4
            'D_25': group['D'].quantile(0.25),     # zusätzlich benötigt für PAC4
            'D_75': group['D'].quantile(0.75)      # zusätzlich benötigt für PAC4
        }
        features.append(feature_row)
        
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Z')
    return X, y


def compute_pacs(X):
    """
    Berechnet aus den aggregierten Features vier Principal Aggregate Components (PACs),
    indem jeweils eine PCA (mit n_components=1) auf die entsprechenden Feature-Gruppen angewendet wird.

    Die Gruppen lauten:
        - PAC1 (Power): P_mean, P_max, P_IQR
        - PAC2 (Force): F_mean, F_max, F_75
        - PAC3 (Distance): D_mean, D_std, D_50
        - PAC4 (Dynamic): D_IQR, D_25, D_75

    Parameter:
        X (DataFrame): DataFrame mit den aggregierten Features.
    Rückgabe:
        pac_df (DataFrame): DataFrame mit den 4 berechneten PACs.
    """
    # PAC1: Power
    power_features = X[['P_mean', 'P_max', 'P_IQR']]
    pca_power = PCA(n_components=1)
    pac1 = pca_power.fit_transform(power_features)
    
    # PAC2: Force
    force_features = X[['F_mean', 'F_max', 'F_75']]
    pca_force = PCA(n_components=1)
    pac2 = pca_force.fit_transform(force_features)
    
    # PAC3: Distance
    distance_features = X[['D_mean', 'D_std', 'D_50']]
    pca_distance = PCA(n_components=1)
    pac3 = pca_distance.fit_transform(distance_features)
    
    # PAC4: Dynamic
    dynamic_features = X[['D_IQR', 'D_25', 'D_75']]
    pca_dynamic = PCA(n_components=1)
    pac4 = pca_dynamic.fit_transform(dynamic_features)
    
    pac_df = pd.DataFrame({
        'PAC1': pac1.flatten(),
        'PAC2': pac2.flatten(),
        'PAC3': pac3.flatten(),
        'PAC4': pac4.flatten()
    }, index=X.index)
    
    return pac_df

def compute_pca(X):
    """
    Berechnet eine einzige PCA mit 4 Komponenten aus den aggregierten Features.
    
    Hier werden alle 12 aggregierten Features (Power, Force, Distance, Dynamic)
    als Input verwendet, und es werden 4 Hauptkomponenten extrahiert.
    
    Parameter:
        X (DataFrame): DataFrame mit den aggregierten Features.
    Rückgabe:
        pac_df (DataFrame): DataFrame mit 4 Spalten, welche die 4 PCA-Komponenten darstellen.
    """
    pca = PCA(n_components=4)
    pac_components = pca.fit_transform(X)
    pac_df = pd.DataFrame(pac_components, columns=["PCA1", "PCA2", "PCA3", "PCA4"], index=X.index)
    return pac_df


# Schritt 1: Aggregation der Daten (mit erweiterter Feature-Berechnung für die PACs)
def prepare_data(data):
    """
    Aggregiert die Prozessdaten je Experiment und berechnet die benötigten Features
    für die Bildung der PACs (Principal Aggregate Components).

    Zusätzlich zu den bereits vorhandenen Features werden folgende aggregierte Größen berechnet:
      - D_mean: Mittelwert von D
      - D_IQR: Interquartilsabstand von D (0.75-Quantil minus 0.25-Quantil)
      - D_25: 25%-Quantil von D
      - D_75: 75%-Quantil von D

    Parameter:
        data (DataFrame): Der vollständige Datensatz mit Spalten wie P, F, D, Z und Experiment_ID.
    Rückgabe:
        X (DataFrame): Aggregierte Eingabe-Features.
        y (Series): Zielvariable Z.
    """
    features = []
    targets = []
    
    for experiment_id, group in data.groupby('Experiment_ID'):
        # Zielvariable: letzter Wert der Spalte Z
        target = group.iloc[-1]['Z']
        targets.append(target)

        feature_row = {
            # Power-Features
            'P_mean': group['P'].mean(),
            'P_max': group['P'].max(),
            'P_IQR': group['P'].quantile(0.75) - group['P'].quantile(0.25),
            
            # Force-Features
            'F_mean': group['F'].mean(),
            'F_max': group['F'].max(),
            'F_75': group['F'].quantile(0.75),
            
            # Distance-Features
            'D_mean': group['D'].mean(),         # zusätzlich benötigt für PAC3
            'D_std': group['D'].std(),
            'D_50': group['D'].median(),
            
            # Dynamic-Features
            'D_IQR': group['D'].quantile(0.75) - group['D'].quantile(0.25),  # zusätzlich benötigt für PAC4
            'D_25': group['D'].quantile(0.25),     # zusätzlich benötigt für PAC4
            'D_75': group['D'].quantile(0.75)      # zusätzlich benötigt für PAC4
        }
        features.append(feature_row)
        
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Z')
    return X, y


def compute_pca(X):
    """
    Berechnet eine einzige PCA mit 4 Komponenten aus den aggregierten Features.
    
    Hier werden alle 12 aggregierten Features (Power, Force, Distance, Dynamic)
    als Input verwendet, und es werden 4 Hauptkomponenten extrahiert.
    
    Parameter:
        X (DataFrame): DataFrame mit den aggregierten Features.
    Rückgabe:
        pac_df (DataFrame): DataFrame mit 4 Spalten, welche die 4 PCA-Komponenten darstellen.
    """
    pca = PCA(n_components=4)
    pac_components = pca.fit_transform(X)
    pac_df = pd.DataFrame(pac_components, columns=["PCA1", "PCA2", "PCA3", "PCA4"], index=X.index)
    return pac_df

# Klasse zur Datenverwaltung
class DataStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Lädt die Daten aus der CSV-Datei."""
        self.data = pd.read_csv(self.file_path)

    def get_data(self):
        """Gibt die geladenen Daten zurück."""
        if self.data is None:
            raise ValueError("Daten wurden noch nicht geladen. Verwende 'load_data', um sie zu laden.")
        return self.data


def save_results_to_csv(results, filename=None):
    """
    Speichert die Ergebnisse der Modelle in einer CSV-Datei.

    Parameter:
      - results: dict, Schlüssel sind Modellnamen, Werte sind Dictionaries mit Modellmetriken.
      - filename: Dateiname der Ausgabe-CSV. Falls None, wird ein Standardname mit Zeitstempel verwendet.
    """
    if filename is None:
        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        filename = f"model_results_{timestamp}.csv"

    data = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Best Params": metrics["best_params"],
            "Overall MAE": metrics["overall_mae"],
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    # Datei mit den Daten (angepasst auf Deine Datenquelle)
    file_path = "../../data/complete_data.csv"
    
    data_storage = DataStorage(file_path)
    data_storage.load_data()
    data = data_storage.get_data()

    # Aggregation der Daten (Features und Zielvariable)
    X, y = prepare_data(data)
    
    # Berechnung der PACs aus den aggregierten Features
    #X_pac = compute_pacs(X)
    X_pac = compute_pca(X)

    # Aufteilen in Trainings- und Testdaten (verwende nun X_pac)
    X_train, X_test, y_train, y_test = train_test_split(X_pac, y, test_size=0.2, random_state=None)
    
    # Skalierung: Nur Trainingsdaten fitten und beide Sätze transformieren
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test),
                                 columns=X_test.columns, index=X_test.index)

    y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
                               name=y_train.name, index=y_train.index)
    y_test_scaled = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(),
                              name=y_test.name, index=y_test.index)

    # Definition der Modelle
    models = {
        "SVR": SVR(kernel='rbf'),
    }

    models_simple = {
        "SVR": SVR(kernel='rbf',
                gamma = 1, 
                epsilon = np.float64(0.004281332398719396), 
                degree = np.int64(2), 
                C = np.float64(12.067926406393289))
    }
    # Hyperparameter-Verteilungen
    param_distributions = {
        "SVR": {
            "C": np.logspace(-3, 2, 50),
            "epsilon": np.logspace(-3, 1, 20),
            "degree": np.arange(2, 5),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        },
    }

    timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    filename = f"model_results_{timestamp}.csv"

    results = evaluation.simple_model_testing_with_logging_normalized(
        models_simple, 
        X_train_scaled,
        y_train_scaled, 
        X_test = X_test_scaled,
        y_test = y_test_scaled,
        target_scaler=scaler_y, 
        #n_iter=50, 
        cv=5, 
        random_state=None,
        save_plots=True
    )
    ## Aufruf der Evaluationsfunktion (hier die angepasste Version, die mit normalisierten Daten arbeitet)
    #results = evaluation.randomized_search_cv_with_logging_normalized(
    #    models, 
    #    X_train_scaled,
    #    y_train_scaled, 
    #    param_distributions,
    #    target_scaler=scaler_y, 
    #    n_iter=50, 
    #    cv=5, 
    #    random_state=None,
    #    scoring='neg_mean_absolute_error',
    #    save_plots=True
    #)

    #save_results_to_csv(results, filename)


if __name__ == "__main__":
    main()
