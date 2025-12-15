import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import evaluation
import data_preperation


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
import joblib

def compute_pca(X, n_components=4):
    """
    Wendet eine PCA auf alle aggregierten Features an und extrahiert
    n_components Hauptkomponenten.

    Parameter:
        X (DataFrame): DataFrame mit den aggregierten Features.
        n_components (int): Anzahl der Hauptkomponenten.
        
    Rückgabe:
        X_pca (DataFrame): DataFrame mit den Hauptkomponenten.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    # Erstelle Spaltennamen: PCA1, PCA2, ... abhängig von n_components
    col_names = [f'PCA{i+1}' for i in range(n_components)]
    X_pca = pd.DataFrame(pca_result, index=X.index, columns=col_names)
    return pca, X_pca


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
    # Datei mit den Daten
    file_path = "../../data/noZeroZ_data.csv"
    
    data_storage = DataStorage(file_path)
    data_storage.load_data()
    data = data_storage.get_data()

    # Aggregation der Daten (Features und Zielvariable)
    X, y = data_preperation.prepare_data(data)
    
    # Berechnung der PCA aus allen Features
    pca, X_pca = compute_pca(X)
    
    # Aufteilen in Trainings- und Testdaten (verwende nun X_pca)
    x_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
    
    # Skalierung: Nur Trainingsdaten fitten und beide Sätze transformieren
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_train_scaled = pd.DataFrame(scaler_x.fit_transform(x_train),
                                  columns=x_train.columns, index=x_train.index)
    X_test_scaled = pd.DataFrame(scaler_x.transform(X_test),
                                 columns=X_test.columns, index=X_test.index)

    y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
                               name=y_train.name, index=y_train.index)
    y_test_scaled = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(),
                              name=y_test.name, index=y_test.index)

    # Definition der Modelle (hier als Beispiel: SVR – ggf. weitere Modelle ergänzen)
    models = {
        "SVR": SVR(kernel='rbf'),
        # Weitere Modelle können hier hinzugefügt werden
    }

    # Hyperparameter-Verteilungen (Beispiel für SVR, weitere Modelle ggf. anpassen)
    param_distributions = {
        "SVR": {
            "C": np.logspace(-3, 2, 50),
            "epsilon": np.logspace(-3, 1, 20),
            "degree": np.arange(2, 5),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        },
    }

    # Aufruf der Evaluationsfunktion (hier die angepasste Version, die mit normalisierten Daten arbeitet)
    results = evaluation.randomized_search_cv_with_logging_normalized(
        models, 
        x_train_scaled,
        y_train_scaled, 
        param_distributions,
        target_scaler=scaler_y, 
        n_iter=50, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        save_plots=True
    )

    # Speichere das beste Modell
    best_model_name = max(results, key=lambda k: results[k]["overall_mae"])
    # Das beste Modell aus den Ergebnissen holen
    best_model = results[best_model_name]["best_model"]  

    model_filename = f"model_{best_model_name}.pkl"
    joblib.dump(best_model, model_filename)

    print(f"Best model ({best_model_name}) saved as {model_filename}")

    joblib.dump(pca, "pca_model.pkl")
    print("PCA model saved as pca_model.pkl")

    joblib.dump(scaler_x, "scaler_x.pkl")
    print("Feature scaler saved as scaler_x.pkl")

    joblib.dump(scaler_y, "scaler_y.pkl")
    print("Target scaler saved as scaler_y.pkl")


if __name__ == "__main__":
    main()
