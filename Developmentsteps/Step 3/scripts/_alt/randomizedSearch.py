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

def randomized_search_cv_with_logging(models, X_train, y_train, param_distributions, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=None):
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
            verbose=2
        )

        search.fit(X_train, y_train)

        # Extract the best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # Perform cross-validated predictions
        y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)

        # Create a DataFrame for analysis
        comparison_df = pd.DataFrame({
            'y_true': y_train,
            'y_pred': y_pred
        })
        comparison_df['Error'] = comparison_df['y_true'] - comparison_df['y_pred']

        # Plot Soll-Ist Vergleich
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name)

        # Log results
        results[model_name] = {
            "best_estimator": best_model,
            "best_params": best_params,
            "best_score": best_score,
            "comparison": comparison_df  # Store actual vs predicted comparison
        }

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best score for {model_name}: {best_score}\n")

    return results

def randomized_search_xTrainNormalized(models, X_train, y_train_original, param_distributions, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=None):
    """
    Perform RandomizedSearchCV on multiple models with detailed logging and visualizations.
    Handles post-hoc rescaling for unnormalized target values (y).

    Parameters:
    - models: dict, keys are model names, values are model instances
    - X_train: Training feature matrix
    - y_train_original: Original (unnormalized) target vector
    - param_distributions: dict, keys are model names, values are parameter distributions for RandomizedSearchCV
    - n_iter: Number of iterations for RandomizedSearchCV (default=50)
    - scoring: Scoring metric for RandomizedSearchCV (default='neg_mean_squared_error')
    - cv: Number of cross-validation folds (default=5)
    - random_state: Random state for reproducibility (default=None)

    Returns:
    - results: dict, keys are model names, values are the best estimator, parameters, and logs
    """
    results = {}

    # Normalize y_train for training purposes
    y_mean = np.mean(y_train_original)
    y_std = np.std(y_train_original)
    y_train_norm = (y_train_original - y_mean) / y_std  # Normalized target

    for model_name, model in models.items():
        print(f"Running RandomizedSearchCV for {model_name}...")

        if model_name not in param_distributions:
            print(f"No parameter distribution found for {model_name}. Skipping...")
            continue

        # Perform RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=8,
            verbose=2
        )

        search.fit(X_train, y_train_norm)  # Use normalized y for training

        # Extract the best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # Perform cross-validated predictions (still normalized)
        y_pred_norm = cross_val_predict(best_model, X_train, y_train_norm, cv=cv)

        # Rescale predictions to original scale
        y_pred_rescaled = (y_pred_norm * y_std) + y_mean

        # Create a DataFrame for analysis
        comparison_df = pd.DataFrame({
            'y_true': y_train_original,  # Original y
            'y_pred': y_pred_rescaled   # Rescaled predictions
        })
        comparison_df['Error'] = comparison_df['y_true'] - comparison_df['y_pred']

        # Plot Soll-Ist Vergleich (assumes function exists)
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name)

        # Log results
        results[model_name] = {
            "best_estimator": best_model,
            "best_params": best_params,
            "best_score": best_score,
            "comparison": comparison_df  # Store actual vs predicted comparison
        }

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best score for {model_name}: {best_score}\n")

    return results

def randomized_search_xTrainNormalized_mae(models, X_train, y_train_original, param_distributions, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=None):
    """
    Perform RandomizedSearchCV on multiple models with detailed logging and visualizations.
    Handles post-hoc rescaling for unnormalized target values (y).

    Parameters:
    - models: dict, keys are model names, values are model instances
    - X_train: Training feature matrix
    - y_train_original: Original (unnormalized) target vector
    - param_distributions: dict, keys are model names, values are parameter distributions for RandomizedSearchCV
    - n_iter: Number of iterations for RandomizedSearchCV (default=50)
    - scoring: Scoring metric for RandomizedSearchCV (default='neg_mean_squared_error')
    - cv: Number of cross-validation folds (default=5)
    - random_state: Random state for reproducibility (default=None)

    Returns:
    - results: dict, keys are model names, values are the best estimator, parameters, and logs
    """
    results = {}

    # Normalize y_train for training purposes
    y_mean = np.mean(y_train_original)
    y_std = np.std(y_train_original)
    y_train_norm = (y_train_original - y_mean) / y_std  # Normalized target
    
    for model_name, model in models.items():
        print(f"Running RandomizedSearchCV for {model_name}...")

        if model_name not in param_distributions:
            print(f"No parameter distribution found for {model_name}. Skipping...")
            continue

        # Perform RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=8,
            verbose=2
        )

        search.fit(X_train, y_train_norm)  # Use normalized y for training

        # Extract the best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # Perform cross-validated predictions (still normalized)
        y_pred_norm = cross_val_predict(best_model, X_train, y_train_norm, cv=cv)

        # Rescale predictions to original scale
        y_pred_rescaled = (y_pred_norm * y_std) + y_mean

        # Calculate MAE for each fold and overall
        fold_mae = []
        for train_idx, test_idx in KFold(n_splits=cv, shuffle=True, random_state=random_state).split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train_norm.iloc[train_idx], y_train_norm.iloc[test_idx]

            best_model.fit(X_train_fold, y_train_fold)
            y_pred_fold_norm = best_model.predict(X_test_fold)
            y_pred_fold_rescaled = (y_pred_fold_norm * y_std) + y_mean

            mae_fold = mean_absolute_error((y_test_fold * y_std) + y_mean, y_pred_fold_rescaled)
            fold_mae.append(mae_fold)

        avg_mae = np.mean(fold_mae)
        print(f"MAE for each fold: {fold_mae}")
        print(f"Average MAE: {avg_mae}")

        # Create a DataFrame for analysis
        comparison_df = pd.DataFrame({
            'y_true': y_train_original,  # Original y
            'y_pred': y_pred_rescaled   # Rescaled predictions
        })
        comparison_df['Error'] = comparison_df['y_true'] - comparison_df['y_pred']

        # Plot Soll-Ist Vergleich (assumes function exists)
        plot_soll_ist(comparison_df['y_true'], comparison_df['y_pred'], model_name)

        # Log results
        results[model_name] = {
            "best_estimator": best_model,
            "best_params": best_params,
            "best_score": best_score,
            "comparison": comparison_df,  # Store actual vs predicted comparison
            "fold_mae": fold_mae,  # Store MAE for each fold
            "average_mae": avg_mae  # Store average MAE
        }

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best score for {model_name}: {best_score}")
        print(f"Average MAE for {model_name}: {avg_mae}\n")

    return results


def plot_soll_ist(y_true, y_pred, model_name):
    """
    Plot the difference between true and predicted values for each iteration.

    Parameters:
    - y_true: Array of true target values
    - y_pred: Array of predicted target values
    - model_name: Name of the model being evaluated
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.title(f"Soll vs. Ist für {model_name}")
    plt.xlabel("Soll-Werte (y_true)")
    plt.ylabel("Ist-Werte (y_pred)")
    plt.legend()
    plt.grid(True)
    plt.show()
