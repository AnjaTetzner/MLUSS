
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import evaluation

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR
from tabulate import tabulate   
import csv
import numpy as np
import pandas as pd

from datetime import datetime

# Schritt 1: Aggregation der Daten (ohne Skalierung)
def prepare_data(data):
    features = []
    targets = []
    
    for experiment_id, group in data.groupby('Experiment_ID'):
        target = group.iloc[-1]['Z']
        targets.append(target)
        
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


def main():
    #file_path = "../../data/complete_data.csv"
    file_path = "../../data/noZeroZ_data.csv"
    
    data_storage = DataStorage(file_path)
    data_storage.load_data()
    data = data_storage.get_data()

    X, y = prepare_data(data)

    # Aufteilen in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skalierung: Nur Trainingsdaten fitten und beide Sätze transformieren
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns, index=X_test.index)

    y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(), 
                               name=y_train.name, index=y_train.index)
    y_test_scaled = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), 
                              name=y_test.name, index=y_test.index)

    models = {
        #"Ridge": Ridge(),
        #"RidgeCV": RidgeCV(),
        #"LassoCV": LassoCV(max_iter=500000),
        #"ElasticNet": ElasticNet(max_iter=100000, tol=1e-3),
        #"SVR": SVR(kernel='linear'),
        #"SVR": SVR(kernel='poly'),
        "SVR": SVR(kernel='rbf'),
        #"LassoCV": LassoCV(max_iter=500000)
    }

    param_distributions = {
        #"Ridge": {
        #    "alpha": np.logspace(-4, 4, 50),
        #},
        "RidgeCV": {
            "cv": [5, 6, 7, 8, 9, 10]
        },
        #"Lasso": {
        #    "alpha": np.logspace(-4, 1, 50),
        #},
         "LassoCV": {  # Entfernen oder in models hinzufügen
             "cv": [5, 6, 7, 8, 9, 10]
         },
        "ElasticNet": {
            "alpha": np.logspace(-4, 1, 50),
            "l1_ratio": np.linspace(0, 1, 20),
        },
        "SVR": {
            "C": np.logspace(-3, 2, 50),
            "epsilon": np.logspace(-3, 1, 20),
            "degree": np.arange(2, 5),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        },
    }

    timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    filename = f"model_results_{timestamp}.csv"

    results = evaluation.randomized_search_cv_with_logging_normalized(
        models, 
        X_train_scaled,
        y_train_scaled, 
        param_distributions,
        target_scaler=scaler_y, 
        n_iter=50, 
        cv=5, 
        random_state=None,
        scoring='neg_mean_absolute_error',
        save_plots=True
    )

    save_results_to_csv(results, filename)

if __name__ == "__main__":
    main()
