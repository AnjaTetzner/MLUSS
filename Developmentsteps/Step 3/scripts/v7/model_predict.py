import pandas as pd
import joblib
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import evaluation
import data_preperation

# Modell und Skalierer laden
def load_model_and_scalers():
    model = joblib.load("model_SVR.pkl")  # Das beste Modell (oder anderes Modell, falls geändert)
    pca = joblib.load("pca_model.pkl")
    scaler_X = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    return model, pca, scaler_X, scaler_y

"""
# Vorhersage für eine neue CSV-Datei treffen
def predict_from_csv(file_path, model, pca, scaler_X, scaler_y, expected_features={"P_mean", "P_max", "P_IQR", "F_mean", "F_max", "F_75", "D_std", "D_50"}):
    # Rohdaten laden
    data = pd.read_csv(file_path)

    # Aggregation der Features mit prepare_data()
    X, y = data_preperation.prepare_data(data)


    # Stelle sicher, dass die Spalten übereinstimmen
    X = X.reindex(columns=expected_features, fill_value=0)
    print(X)
    # Konvertiere `X` in ein NumPy-Array, um Fehler mit pandas zu vermeiden
    X_np = X.to_numpy()
    print(f"First rows of X:\n{X_np[:5]}")
    exit()

    # Wende die gespeicherte PCA-Transformation an
    X_pca = pca.transform(X_np)

    # Skalierung anwenden
    X_scaled = scaler_X.transform(X_pca)

    # Vorhersage treffen
    predictions_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Ergebnisse speichern
    result_df = pd.DataFrame({"Experiment_ID": data["Experiment_ID"].unique(), "Predicted_Z": predictions})
    result_file = file_path.replace(".csv", "_predictions.csv")
    result_df.to_csv(result_file, index=False)
    print(f"Predictions saved to {result_file}")
"""

# Vorhersage für eine neue CSV-Datei treffen
def predict_from_csv(file_path, model, pca, scaler_X, scaler_y):
    # Rohdaten laden
    data = pd.read_csv(file_path)

    # Aggregation der Features mit prepare_data()
    X, _ = data_preperation.prepare_data(data)  # Zielwert Z ignorieren

    # Wende die gespeicherte PCA-Transformation an (und behalte Spaltennamen!)
    X_pca = pd.DataFrame(pca.transform(X), columns=[f"PCA{i+1}" for i in range(pca.n_components_)])

    # Skalierung anwenden (und Spaltennamen beibehalten)
    X_scaled = pd.DataFrame(scaler_X.transform(X_pca), columns=X_pca.columns)

    # Vorhersage treffen
    predictions_scaled = model.predict(X_scaled) 
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Ergebnisse speichern
    result_df = pd.DataFrame({"Experiment_ID": data["Experiment_ID"].unique(), "Predicted_Z": predictions})
    print(result_df)

if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #    print("Usage: python predict_z_values.py <csv_file>")
    #    sys.exit(1)
    
    file_path = "../../data/prediction_test/experiment_1575.csv"
    model, pca, scaler_X, scaler_y = load_model_and_scalers()
    predict_from_csv(file_path, model, pca, scaler_X, scaler_y)