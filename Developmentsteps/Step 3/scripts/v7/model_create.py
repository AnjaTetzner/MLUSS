import sys
sys.path.insert(1, '../')

import evaluation
import data_preperation

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def compute_pca(X, n_components=4):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    col_names = [f'PCA{i+1}' for i in range(n_components)]
    X_pca = pd.DataFrame(pca_result, index=X.index, columns=col_names)
    return pca, X_pca

class DataStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def get_data(self):
        if self.data is None:
            raise ValueError("Daten wurden noch nicht geladen. Verwende 'load_data', um sie zu laden.")
        return self.data

def main():
    file_path = "../../data/noZeroZ_data.csv"
    data_storage = DataStorage(file_path)
    data_storage.load_data()
    data = data_storage.get_data()

    # Regression
    X_reg, y_reg = data_preperation.prepare_data(data)
    pca_reg, X_pca_reg = compute_pca(X_reg)
    x_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_pca_reg, y_reg, test_size=0.2, random_state=42)
    
    scaler_x_reg = MinMaxScaler()
    scaler_y_reg = MinMaxScaler()
    x_train_scaled_reg = pd.DataFrame(scaler_x_reg.fit_transform(x_train_reg), columns=x_train_reg.columns, index=x_train_reg.index)
    X_test_scaled_reg = pd.DataFrame(scaler_x_reg.transform(X_test_reg), columns=X_test_reg.columns, index=X_test_reg.index)
    y_train_scaled_reg = pd.Series(scaler_y_reg.fit_transform(y_train_reg.values.reshape(-1, 1)).flatten(), index=y_train_reg.index)
    y_test_scaled_reg = pd.Series(scaler_y_reg.transform(y_test_reg.values.reshape(-1, 1)).flatten(), index=y_test_reg.index)
    
    models_reg = {"SVR": SVR(kernel='rbf'), "LinearRegression": LinearRegression()}
    
    param_distributions_reg = {
        "SVR": {
            "C": np.logspace(-3, 2, 50),
            "epsilon": np.logspace(-3, 1, 20),
            "gamma": ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    }
    
    results_reg = evaluation.randomized_search_cv_with_logging_normalized(
        models_reg, 
        x_train_scaled_reg,
        y_train_scaled_reg, 
        param_distributions_reg,
        target_scaler=scaler_y_reg, 
        n_iter=50, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        save_plots=True
    )
    
    best_model_name_reg = max(results_reg, key=lambda k: results_reg[k]["overall_mae"], default=None)
    if best_model_name_reg:
        best_model_reg = results_reg[best_model_name_reg]["best_model"]  
        joblib.dump(best_model_reg, f"models/model_{best_model_name_reg}_reg.pkl")
    
    joblib.dump(pca_reg, "models/pca_model_reg.pkl")
    joblib.dump(scaler_x_reg, "models/scaler_x_reg.pkl")
    joblib.dump(scaler_y_reg, "models/scaler_y_reg.pkl")
    
    # Klassifikation
    X_cls, y_cls = data_preperation.prepare_data_classification(data)
    label_encoder = LabelEncoder()
    y_encoded_cls = label_encoder.fit_transform(y_cls)
    
    pca_cls, X_pca_cls = compute_pca(X_cls)
    x_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_pca_cls, y_encoded_cls, test_size=0.2, random_state=42)
    
    scaler_x_cls = MinMaxScaler()
    x_train_scaled_cls = pd.DataFrame(scaler_x_cls.fit_transform(x_train_cls), columns=x_train_cls.columns, index=x_train_cls.index)
    X_test_scaled_cls = pd.DataFrame(scaler_x_cls.transform(X_test_cls), columns=X_test_cls.columns, index=X_test_cls.index)
    
    # y_train_cls als pandas Series umwandeln
    y_train_cls = pd.Series(y_train_cls, index=x_train_cls.index)
    
    models_cls = {"SVC": SVC(kernel='rbf')}
    param_distributions_cls = {
        "SVC": {
            "C": np.logspace(-3, 2, 50),
            "gamma": ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            "kernel": ['rbf', 'poly', 'sigmoid']
        }
    }
    
    results_cls = evaluation.randomized_search_cv_with_logging_classification(
        models_cls, 
        x_train_scaled_cls,
        y_train_cls, 
        param_distributions_cls,
        n_iter=50, 
        cv=5, 
        scoring='accuracy',
        save_plots=True
    )
    
    best_model_name_cls = max(results_cls, key=lambda k: results_cls[k]["accuracy"], default=None)
    if best_model_name_cls:
        best_model_cls = results_cls[best_model_name_cls]["best_model"]  
        joblib.dump(best_model_cls, f"models/model_{best_model_name_cls}_cls.pkl")
    
    joblib.dump(pca_cls, "models/pca_model_cls.pkl")
    joblib.dump(scaler_x_cls, "models/scaler_x_cls.pkl")
    joblib.dump(label_encoder, "models/label_encoder_cls.pkl")
    
    print("Regression- und Klassifikationsmodelle wurden erfolgreich gespeichert.")

if __name__ == "__main__":
    main()
