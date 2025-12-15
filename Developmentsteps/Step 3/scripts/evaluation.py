from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

def randomized_search_cv_with_logging_normalized(models, 
                                                X_train, 
                                                y_train, 
                                                param_distributions, 
                                                target_scaler, 
                                                n_iter=50, 
                                                scoring='neg_mean_absolute_error', 
                                                cv=5, 
                                                random_state=None, 
                                                save_plots=False):
    """
    Führt RandomizedSearchCV für mehrere Modelle mit detailliertem Logging und Visualisierungen durch.
    Es wird davon ausgegangen, dass die übergebenen Daten bereits mittels Min‑Max‑Skalierung normalisiert wurden.
    Die Vorhersagen werden anschließend in die Ursprungsform zurücktransformiert.
    
    Parameter:
        models (dict): Schlüssel sind Modellnamen, Werte sind Modellinstanzen.
        X_train (DataFrame): Trainingsfeatures (normalisiert).
        y_train (Series): Trainingszielvariable (normalisiert).
        param_distributions (dict): Schlüssel sind Modellnamen, Werte sind Parameterverteilungen für RandomizedSearchCV.
        target_scaler (MinMaxScaler): Skalierer, der auf die Zielvariable angewendet wurde (wichtig für die Rücktransformation).
        n_iter (int): Anzahl der Iterationen für RandomizedSearchCV (default=50).
        scoring (str): Bewertungsmetrik für RandomizedSearchCV (default='neg_mean_absolute_error').
        cv (int): Anzahl der Cross-Validation-Folds (default=5).
        random_state (int oder None): Zufallsstatus für Reproduzierbarkeit (default=None).
        save_plots (bool): Wenn True, werden Plots gespeichert statt angezeigt (default=False).
    
    Rückgabe:
        results (dict): Schlüssel sind Modellnamen, Werte sind Dictionaries mit dem besten Estimator, den besten Parametern, 
                        Metriken (AVG MAE und Overall MAE in der Ursprungsform) und einem Vergleichs-DataFrame.
    """

    results = {}

    for model_name, model in models.items():
        print(f"Running RandomizedSearchCV for {model_name}...")

        if model_name not in param_distributions:
            print(f"No parameter distribution found for {model_name}. Skipping...")
            continue

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=8,
            verbose=0
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Cross-validation: Berechnung des MAE in Originalskala
        fold_mae = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_test_fold = X_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]

            best_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = best_model.predict(X_test_fold)

            # Rücktransformation in die Ursprungsform
            y_pred_fold_original = target_scaler.inverse_transform(y_pred_fold.reshape(-1, 1)).flatten()
            y_test_fold_original = target_scaler.inverse_transform(y_test_fold.values.reshape(-1, 1)).flatten()

            mae_fold = mean_absolute_error(y_test_fold_original, y_pred_fold_original)
            fold_mae.append(mae_fold)

        avg_mae = np.mean(fold_mae)

        # Gesamte cross-validated Vorhersagen
        y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)
        y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_original = target_scaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten()
        overall_mae = mean_absolute_error(y_true_original, y_pred_original)

        # DataFrame zum Vergleich (Soll-Ist)
        comparison_df = pd.DataFrame({
            'y_true': y_true_original,
            'y_pred': y_pred_original
        })
        comparison_df['Error'] = comparison_df['y_true'] - comparison_df['y_pred']

        results[model_name] = {
            "best_model": best_model,
            "best_params": best_params,
            "avg_mae": avg_mae,
            "overall_mae": overall_mae,
            "comparison": comparison_df
        }

        print("=========================")
        print(f"Model {model_name}:")
        print(f"Parameters: {best_params}")
        print(f"AVG MAE (original scale): {avg_mae}")
        print(f"Overall MAE (original scale): {overall_mae}")

        # Annahme: Es gibt eine Funktion 'plot_soll_ist' zum Plotten des Soll-Ist-Vergleichs.
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name, overall_mae, save_plot=save_plots)

    return results


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def randomized_search_cv_with_logging_classification(models, 
                                                      X_train, 
                                                      y_train, 
                                                      param_distributions, 
                                                      n_iter=50, 
                                                      scoring='accuracy', 
                                                      cv=5, 
                                                      random_state=None, 
                                                      save_plots=False):
    """
    Führt RandomizedSearchCV für mehrere Klassifikationsmodelle mit detailliertem Logging und Visualisierungen durch.
    
    Parameter:
        models (dict): Schlüssel sind Modellnamen, Werte sind Modellinstanzen.
        X_train (DataFrame): Trainingsfeatures (normalisiert).
        y_train (Series oder Array): Trainingszielvariable.
        param_distributions (dict): Schlüssel sind Modellnamen, Werte sind Parameterverteilungen für RandomizedSearchCV.
        n_iter (int): Anzahl der Iterationen für RandomizedSearchCV (default=50).
        scoring (str): Bewertungsmetrik für RandomizedSearchCV (default='accuracy').
        cv (int): Anzahl der Cross-Validation-Folds (default=5).
        random_state (int oder None): Zufallsstatus für Reproduzierbarkeit (default=None).
        save_plots (bool): Wenn True, werden Plots gespeichert statt angezeigt (default=False).
    
    Rückgabe:
        results (dict): Schlüssel sind Modellnamen, Werte sind Dictionaries mit dem besten Estimator, den besten Parametern, 
                        Metriken (Accuracy) und einer Konfusionsmatrix.
    """

    results = {}

    for model_name, model in models.items():
        print(f"Running RandomizedSearchCV for {model_name}...")

        if model_name not in param_distributions:
            print(f"No parameter distribution found for {model_name}. Skipping...")
            continue

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=8,
            verbose=0
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

        # Cross-validation: Berechnung der Accuracy
        y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_train, y_pred)

        results[model_name] = {
            "best_model": best_model,
            "best_params": best_params,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }

        print("=========================")
        print(f"Model {model_name}:")
        print(f"Parameters: {best_params}")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(pd.DataFrame(report).transpose())
        print("Confusion Matrix:")
        print(conf_matrix)

        if save_plots:
            plt.figure(figsize=(6, 5))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.colorbar()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(f'confusion_matrix_{model_name}.png')
            plt.close()
        
    return results

def randomized_search_cv_with_logging(models, X_train, y_train, param_distributions, n_iter=50, scoring='neg_mean_absolute_error', cv=5, random_state=None, save_plots=False):
    """
    Perform RandomizedSearchCV on multiple models with detailed logging and visualizations.

    Parameters:
    - models: dict, keys are model names, values are model instances
    - X_train: Training feature matrix
    - y_train: Training target vector
    - param_distributions: dict, keys are model names, values are parameter distributions for RandomizedSearchCV
    - n_iter: Number of iterations for RandomizedSearchCV (default=50)
    - scoring: Scoring metric for RandomizedSearchCV (default='neg_mean_absolute_error')
    - cv: Number of cross-validation folds (default=5)
    - random_state: Random state for reproducibility (default=None)
    - save_plots: If True, save plots instead of displaying them (default=False)

    Returns:
    - results: dict, keys are model names, values are the best estimator, parameters, and logs
    """
    results = {}

    for model_name, model in models.items():
        print(f"Running RandomizedSearchCV for {model_name}...")

        if model_name not in param_distributions:
            print(f"No parameter distribution found for {model_name}. Skipping...")
            continue

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=8,
            verbose=0
        )

        search.fit(X_train, y_train)

        # Extract the best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_mae = search.best_score_

        # Perform cross-validated predictions
        fold_mae = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

            best_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = best_model.predict(X_test_fold)

            mae_fold = mean_absolute_error(y_test_fold, y_pred_fold)
            fold_mae.append(mae_fold)

        avg_mae = np.mean(fold_mae)

        # Perform overall cross-validated predictions
        y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)

        # Create a DataFrame for analysis
        comparison_df = pd.DataFrame({
            'y_true': y_train,
            'y_pred': y_pred
        })
        comparison_df['Error'] = comparison_df['y_true'] - comparison_df['y_pred']

        # Log results
        results[model_name] = {
            "best_estimator": best_model,
            "best_params": best_params,
            #"avg_mae": avg_mae,        # Add average MAE to the results
            "overall_mae": best_mae,        # Add overall MAE to the results
            "comparison": comparison_df  # Store actual vs predicted comparison
        }

        print("=========================")
        print(f"Model {model_name}:")
        print(f"parameters: {best_params}")
        print(f"AVG MAE: {avg_mae}")
        print(f"Best MAE: {best_mae}")

        # Plot Soll-Ist Vergleich
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name, best_mae, save_plot=save_plots)

    return results

def simple_model_testing_with_logging(models, X_train, y_train, n_iter=50, cv=5, random_state=None, save_plots=False):
    """
    Test models with fixed parameters and log results, maintaining a loop to compute averages for MAE and MSE.

    Parameters:
    - models: dict, keys are model names, values are model instances
    - X_train: Training feature matrix
    - y_train: Training target vector
    - param_distributions: dict, keys are model names, values are fixed parameters for testing
    - n_iter: Number of iterations for testing (default=50)
    - scoring: Scoring metric (default='neg_mean_squared_error')
    - cv: Number of cross-validation folds (default=5)
    - random_state: Random state for reproducibility (default=None)
    - save_plots: If True, save plots instead of displaying them (default=False)

    Returns:
    - results: dict, keys are model names, values are the best estimator, parameters, and logs
    """
    results = {}

    for model_name, model_function in models.items():
        print(f"Testing fixed parameters for {model_name}...")

        model = model_function

        # Perform overall cross-validated predictions
        y_pred = cross_val_predict(model, X_train, y_train, cv=cv)

        # Calculate overall MAE and MSE
        overall_mae = mean_absolute_error(y_train, y_pred)

        # Create a DataFrame for analysis
        comparison_df = pd.DataFrame({
            'y_true': y_train,
            'y_pred': y_pred
        })
        comparison_df['Error'] = comparison_df['y_true'] - comparison_df['y_pred']

        # Log results
        results[model_name] = {
            "overall_mae": overall_mae,
            "best_params": "none",
            "comparison": comparison_df
        }

        print("=========================")
        print(f"Model {model_name}:")
        print(f"Overall MAE: {overall_mae}")

        # Plot Soll-Ist Vergleich
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name, overall_mae, save_plot=save_plots)

    return results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import cross_val_predict

def plot_confusion_matrix(cm, model_name, save_plot=False, cmap=plt.cm.Blues):
    """
    Zeichnet und zeigt (oder speichert) eine Konfusionsmatrix.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'Confusion Matrix (binned) - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.ylabel('True bin')
    plt.xlabel('Predicted bin')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    if save_plot:
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()
    else:
        plt.show()

def simple_model_testing_with_logging_normalized(models, 
                                                 X_train, 
                                                 y_train, 
                                                 X_test, 
                                                 y_test, 
                                                 target_scaler=None, 
                                                 cv=5, 
                                                 random_state=None, 
                                                 save_plots=False,
                                                 num_bins=10):
    """
    Testet Regressionsmodelle mit festen Parametern, loggt Ergebnisse und erstellt zusätzlich
    eine "Konfusionsmatrix", indem die kontinuierlichen Vorhersagen in Kategorien (Bins) unterteilt werden.
    
    Es wird davon ausgegangen, dass die Features bereits normalisiert sind. Falls ein target_scaler 
    (z. B. MinMaxScaler) angegeben ist, werden die Vorhersagen und Zielwerte in ihre Ursprungsform zurücktransformiert.
    
    Parameter:
        models (dict): Schlüssel sind Modellnamen, Werte sind Modellinstanzen (z. B. Regressoren).
        X_train (DataFrame): Trainingsfeatures (normalisiert).
        y_train (Series): Trainingsziel (normalisiert, falls target_scaler angegeben ist).
        X_test (DataFrame): Testfeatures (normalisiert).
        y_test (Series): Testziel (normalisiert, falls target_scaler angegeben ist).
        target_scaler: Skalierer, der für die Zielvariable verwendet wurde (z. B. MinMaxScaler). 
                       Falls angegeben, werden Vorhersagen und Zielwerte zurücktransformiert.
        cv (int): Anzahl der Cross-Validation-Folds (default=5).
        random_state: Zufallsstatus für Reproduzierbarkeit (default=None).
        save_plots (bool): Falls True, werden Plots gespeichert statt angezeigt (default=False).
        num_bins (int): Anzahl der Kategorien (Bins) zur Erstellung der Konfusionsmatrix (default=10).
    
    Rückgabe:
        results (dict): Pro Modell ein Dictionary mit:
            - "train_mae": MAE (CV) auf den Trainingsdaten.
            - "train_mse": MSE (CV) auf den Trainingsdaten.
            - "train_r2": R² (CV) auf den Trainingsdaten.
            - "test_mae": MAE auf den Testdaten.
            - "test_mse": MSE auf den Testdaten.
            - "test_r2": R² auf den Testdaten.
            - "confusion_matrix": Konfusionsmatrix der Testdaten (auf Basis der Bins).
            - "test_comparison": DataFrame mit den tatsächlichen und vorhergesagten Werten im Testdatensatz.
    """
    results = {}

    for model_name, model in models.items():
        print(f"Testing fixed parameters for {model_name}...")

        # Erzeuge CV-Vorhersagen auf den Trainingsdaten
        #y_train_pred = cross_val_predict(model, X_train, y_train, cv=cv, random_state=random_state)
        y_train_pred = cross_val_predict(model, X_train, y_train, cv=cv)
        
        # Rücktransformiere in die Ursprungsform, falls ein Skalierer vorhanden ist
        if target_scaler is not None:
            y_train_pred_orig = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
            y_train_orig = target_scaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten()
        else:
            y_train_pred_orig = y_train_pred
            y_train_orig = y_train.values if hasattr(y_train, 'values') else y_train

        train_mae = mean_absolute_error(y_train_orig, y_train_pred_orig)
        train_mse = mean_squared_error(y_train_orig, y_train_pred_orig)
        train_r2  = r2_score(y_train_orig, y_train_pred_orig)

        # Finales Training auf dem gesamten Trainingsdatensatz und Testvorhersage
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        if target_scaler is not None:
            y_test_pred_orig = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            y_test_orig = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        else:
            y_test_pred_orig = y_test_pred
            y_test_orig = y_test.values if hasattr(y_test, 'values') else y_test

        test_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
        test_mse = mean_squared_error(y_test_orig, y_test_pred_orig)
        test_r2  = r2_score(y_test_orig, y_test_pred_orig)

        # Erstelle eine "Konfusionsmatrix" durch Einteilen der kontinuierlichen Werte in Bins.
        # Die Bins werden basierend auf dem Bereich der Test-Zielwerte festgelegt.
        bins = np.linspace(y_test_orig.min(), y_test_orig.max(), num_bins + 1)
        y_test_binned = np.digitize(y_test_orig, bins)
        y_pred_binned = np.digitize(y_test_pred_orig, bins)
        conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)

        # Plot der Konfusionsmatrix (auf Basis der Bins)
        plot_confusion_matrix(conf_matrix, model_name, save_plot=save_plots)

        # Erstelle einen Vergleichs-DataFrame für den Testdatensatz
        test_comparison_df = pd.DataFrame({
            'y_true': y_test_orig,
            'y_pred': y_test_pred_orig,
            'error': y_test_orig - y_test_pred_orig
        })

        results[model_name] = {
            "train_mae": train_mae,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "confusion_matrix": conf_matrix,
            "test_comparison": test_comparison_df
        }

        print("=========================")
        print(f"Model {model_name}:")
        print(f"Training MAE: {train_mae}")
        print(f"Training MSE: {train_mse}")
        print(f"Training R^2:  {train_r2}")
        print(f"Test MAE: {test_mae}")
        print(f"Test MSE: {test_mse}")
        print(f"Test R^2:  {test_r2}")
        print("Test Confusion Matrix (binned):")
        print(conf_matrix)

    return results


def plot_soll_ist(y_true, y_pred, model_name, mae, save_plot=False):
    """
    Plot the difference between true and predicted values for each iteration.

    Parameters:
    - y_true: Array of true target values
    - y_pred: Array of predicted target values
    - model_name: Name of the model being evaluated
    - mae: Mean Absolute Error to display in the plot
    - save_plot: If True, save the plot to a file instead of displaying it (default=False)
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.title(f"Vorhersage für Model: {model_name}\nMAE: {mae:.2f}")
    plt.xlabel("Soll-Werte (y_true)")
    plt.ylabel("Ist-Werte (y_pred)")
    plt.legend()
    plt.grid(True)

    if save_plot:
        plot_filename = f"soll_ist_{model_name}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    else:
        plt.show()
    plt.close()


#--------------
# Testen der Funktionen

if __name__ == "__main__":
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

    # Beispiel-Daten vorbereiten
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 5), columns=[f"Feature_{i}" for i in range(5)])
    y = pd.Series(np.random.rand(200))

    # Trainings- und Testdaten splitten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelle definieren
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42)
    }

    # Parameter-Distribution (nur zur Konsistenz, aber nicht genutzt in der vereinfachten Version)
    param_distributions = {
        "LinearRegression": {},
        "RandomForestRegressor": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10]
        }
    }

    # Funktion mit RandomizedSearchCV testen
    print("== RandomizedSearchCV Funktion ==")
    results_randomized = randomized_search_cv_with_logging(
        models=models,
        X_train=X_train,
        y_train=y_train,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42,
        save_plots=False
    )

    # Funktion mit einfacher Modellbewertung testen
    print("\n== Simple Testing Funktion ==")
    results_simple = simple_model_testing_with_logging(
        models=models,
        X_train=X_train,
        y_train=y_train,
        n_iter=10,
        cv=5,
        random_state=42,
        save_plots=False
    )

    # Konsistenzprüfung für jedes Modell
#    for model_name in models.keys():
#        print(f"\nPrüfung der Konsistenz für {model_name}...")
#        avg_mse_randomized = results_randomized[model_name]["mse"]
#        overall_mse_randomized = results_randomized[model_name]["overall_mse"]
#        avg_mse_simple = results_simple[model_name]["avg_mse"]
#        overall_mse_simple = results_simple[model_name]["overall_mse"]
#
#        print(f"RandomizedSearch - avg_mse: {avg_mse_randomized}, overall_mse: {overall_mse_randomized}")
#        print(f"Simple Testing   - avg_mse: {avg_mse_simple}, overall_mse: {overall_mse_simple}")
#
#        assert np.isclose(avg_mse_randomized, overall_mse_randomized, atol=1e-5), \
#            f"RandomizedSearch: avg_mse und overall_mse weichen zu stark ab für {model_name}!"
#        assert np.isclose(avg_mse_simple, overall_mse_simple, atol=1e-5), \
#            f"Simple Testing: avg_mse und overall_mse weichen zu stark ab für {model_name}!"
#        print("==> Konsistenz zwischen avg_mse und overall_mse ist gegeben.")