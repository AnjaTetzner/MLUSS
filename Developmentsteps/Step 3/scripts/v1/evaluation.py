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

def randomized_search_cv_with_logging(models, X_train, y_train, param_distributions, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=None, save_plots=False):
    """
    Perform RandomizedSearchCV on multiple models with detailed logging and visualizations.

    Parameters:
    - models: dict, keys are model names, values are model instances
    - X_train: Training feature matrix
    - y_train: Training target vector
    - param_distributions: dict, keys are model names, values are parameter distributions for RandomizedSearchCV
    - n_iter: Number of iterations for RandomizedSearchCV (default=50)
    - scoring: Scoring metric for RandomizedSearchCV (default='neg_mean_squared_error')
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
        best_score = search.best_score_

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

        # Calculate overall MAE
        mae = mean_absolute_error(y_train, y_pred)

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
            "best_score": best_score,
            "mae_per_fold": fold_mae,  # Add per-fold MAE to the results
            "avg_mae": avg_mae,        # Add average MAE to the results
            "overall_mae": mae,        # Add overall MAE to the results
            "comparison": comparison_df  # Store actual vs predicted comparison
        }

        print("=========================")
        print(f"Model {model_name}:")
        print(f"parameters: {best_params}")
        print(f"MSE for parameters: {best_score}")
        print(f"Average MAE: {avg_mae}")

        # Plot Soll-Ist Vergleich
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name, avg_mae, save_plot=True)
       # save_results_to_csv(results)

    return results

def plot_soll_ist(y_true, y_pred, model_name, avg_mae, save_plot=False):
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
    plt.title(f"Vorhersage für Model: {model_name}\nAvg. MAE: {avg_mae:.2f}")
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
