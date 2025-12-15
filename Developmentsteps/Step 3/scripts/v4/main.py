
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import evaluation

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from tabulate import tabulate   
import csv
import numpy as np
import pandas as pd

from datetime import datetime

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

        # Zielvariable Z aus der letzten Zeile ziehen
        target = group.iloc[-1]['Z']
        targets.append(target)
        
        # Aggregation der Eingabefeatures
        feature_row = {
            'P_mean': group['P'].mean(),
            'F_max': group['F'].max(),
            'D_std': group['D'].std(),
            'P_IQR': group['P'].quantile(0.75) - group['P'].quantile(0.25),
            'F_75': group['F'].quantile(0.75),
            'D_50': group['D'].median(),
            'P_max': group['P'].max(),
            'F_mean': group['F'].mean()
        }

        features.append(feature_row)
    
    # Features und Zielvariable in DataFrames umwandeln
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Z')
    
    return X, y
    
# Definiere eine Klasse, um die Daten zu verwalten
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
    Save the results of the models to a CSV file.

    Parameters:
    - results: dict, keys are model names, values are dictionaries containing model metrics and logs
    - filename: Name of the output CSV file. If None, a default name with a timestamp is used.
    """

    # Generate default filename with ISO timestamp if none is provided
    if filename is None:
        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        filename = f"model_results_{timestamp}.csv"

    # Prepare data for saving
    data = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Best Params": metrics["best_params"],
            "Overall MAE": metrics["overall_mae"],
        }
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# Verwendung der Klasse
if __name__ == "__main__":
    file_path ="../../data/complete_data.csv"

    data_storage = DataStorage(file_path)

    # Daten laden
    data_storage.load_data()

    # Retrieve loaded data
    data = data_storage.get_data()

    #-------------------
    # Prepare data
    X, y = prepare_data(data)

   # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Ridge": Ridge(),
        "RidgeCV": RidgeCV(),
        "Lasso": Lasso(max_iter=500000),
        "ElasticNet": ElasticNet(max_iter=500000),
        "SVR": SVR(kernel='linear'),
    }

    models_simple = {
        "LinearRegression": LinearRegression(),
#        "Ridge": Ridge(alpha=0.0001),
#        "Lasso": Lasso(max_iter=5000, alpha=0.0001),
#        "ElasticNet": ElasticNet(alpha=0.0001, max_iter=10000,l1_ratio=0.7368421052631579),
#        "DecisionTreeRegressor": DecisionTreeRegressor(min_samples_split=9, min_samples_leaf=13, max_depth=14),
#        "RandomForest": RandomForestRegressor(n_jobs=8,n_estimators=75, min_samples_split=2, min_samples_leaf=2, max_features=None, max_depth=13),
#        "GradientBoosting": GradientBoostingRegressor(n_estimators=175, min_samples_split=8, min_samples_leaf=7, max_depth=4, learning_rate=0.05455594781168517),
#        "SVR": SVR(kernel='linear', gamma='auto', epsilon=6.158482110660261, degree=3, C=24.420530945486497),
#        "KNeighborsRegressor": KNeighborsRegressor(weights='uniform', p=1, n_neighbors=8),
#        "MLPRegressor": MLPRegressor(max_iter=10000, learning_rate_init=0.00042813323987193956, hidden_layer_sizes=(50, 50), alpha=0.009102981779915217, activation='relu')
    }
    
    param_distributions = {
        
        "Ridge": {
            "alpha": np.logspace(-4, 4, 50),  # Ridge-Regularisierung
        },
        
        "RidgeCV": {
            "cv": [5,6,7,8,9,10]  # Ridge-Regularisierung
        },

        "Lasso": {
            "alpha": np.logspace(-4, 1, 50),  # Lasso-Regularisierung
        },
        
        "LassoCV": {
            "cv": [5,6,7,8,9,10],  # Ridge-Regularisierung
        },

        "ElasticNet": {
            "alpha": np.logspace(-4, 1, 50),  # Regularisierung
            "l1_ratio": np.linspace(0, 1, 20),  # Mischung zwischen L1 und L2
        },
        
        "SVR": {
            "C": np.logspace(-3, 2, 50),  # Regularisierung
            "epsilon": np.logspace(-3, 1, 20),  # Epsilon im ε-SVR
           # "kernel": ["linear", "poly", "rbf"],  # Verschiedene Kernel
            "degree": np.arange(2, 5),  # Nur für Poly-Kernel
            "gamma": ["scale", "auto"],  # Gamma-Parameter
        },
    }

    timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    filename = f"model_results_{timestamp}.csv"

    #Assuming X_train and y_train are defined
    results = evaluation.randomized_search_cv_with_logging(models, X_train, y_train, param_distributions,
              scoring='neg_mean_absolute_error', save_plots=True)

#    results = evaluation.simple_model_testing_with_logging(models_simple, X_train, y_train, n_iter=1, save_plots=True)       
            

    save_results_to_csv(results)
