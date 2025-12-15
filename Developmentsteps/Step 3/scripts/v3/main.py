import evaluation

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
            'P_min': group['P'].min(),
            'P_max': group['P'].max(),
            'F_mean': group['F'].mean(),
            'F_min': group['F'].min(),
            'F_max': group['F'].max(),
            'D_mean': group['D'].mean(),
            'D_min': group['D'].min(),
            'D_max': group['D'].max(),
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
    file_path ="../../data/noZeroZ_data.csv"

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
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=100000),
        "ElasticNet": ElasticNet(max_iter=100000),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_jobs=8),
        "GradientBoosting": GradientBoostingRegressor(),
        "SVR": SVR(kernel='linear'),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "MLPRegressor": MLPRegressor(max_iter=100000)
    }
    
    param_distributions = {
        "LinearRegression": {},  # Keine Hyperparameter zum Tuning
        
        "Ridge": {
            "alpha": np.logspace(-4, 4, 50),  # Ridge-Regularisierung
        },
        
        "Lasso": {
            "alpha": np.logspace(-4, 1, 50),  # Lasso-Regularisierung
        },
        
        "ElasticNet": {
            "alpha": np.logspace(-4, 1, 50),  # Regularisierung
            "l1_ratio": np.linspace(0, 1, 20),  # Mischung zwischen L1 und L2
        },
        
        "DecisionTreeRegressor": {
            "max_depth": np.arange(1, 20),  # Maximale Tiefe des Baums
            "min_samples_split": np.arange(2, 20),  # Minimaler Split
            "min_samples_leaf": np.arange(1, 20),  # Minimal erlaubte Blätter
        },
        
        "RandomForest": {
            "n_estimators": np.arange(50, 200, 25),  # Anzahl der Bäume
            "max_depth": np.arange(5, 20),  # Maximale Tiefe
            "min_samples_split": np.arange(2, 10),  # Minimaler Split
            "min_samples_leaf": np.arange(1, 10),  # Minimal erlaubte Blätter
            "max_features": ["sqrt", "log2", None],  # Anzahl der Features pro Split
        },
        
        "GradientBoosting": {
            "n_estimators": np.arange(50, 200, 25),  # Anzahl der Boosting-Stufen
            "learning_rate": np.logspace(-3, 0, 20),  # Lernrate
            "max_depth": np.arange(3, 10),  # Maximale Tiefe
            "min_samples_split": np.arange(2, 10),  # Minimaler Split
            "min_samples_leaf": np.arange(1, 10),  # Minimal erlaubte Blätter
        },
        
        "SVR": {
            "C": np.logspace(-3, 2, 50),  # Regularisierung
            "epsilon": np.logspace(-3, 1, 20),  # Epsilon im ε-SVR
           # "kernel": ["linear", "poly", "rbf"],  # Verschiedene Kernel
            "degree": np.arange(2, 5),  # Nur für Poly-Kernel
            "gamma": ["scale", "auto"],  # Gamma-Parameter
        },
        
        "KNeighborsRegressor": {
            "n_neighbors": np.arange(1, 30),  # Anzahl der Nachbarn
            "weights": ["uniform", "distance"],  # Gewichtung der Nachbarn
            "p": [1, 2],  # Minkowski-Metrik (1 = Manhattan, 2 = Euklidisch)
        },
        
        "MLPRegressor": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],  # Verschiedene Architekturen
            "alpha": np.logspace(-4, 0, 50),  # L2-Regularisierung
            "learning_rate_init": np.logspace(-4, -1, 20),  # Lernrate
            "activation": ["relu", "tanh", "logistic"],  # Aktivierungsfunktionen
        }
    }

    timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    filename = f"model_results_{timestamp}.csv"

    #Assuming X_train and y_train are defined
    results = evaluation.randomized_search_cv_with_logging(models, X_train, y_train, param_distributions,
              scoring='neg_mean_absolute_error', save_plots=True)

    save_results_to_csv(results)
