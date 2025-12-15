import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA


class LoadTs:
    def __init__(self):
        self.df = pd.DataFrame()
        self.label_encoder = LabelEncoder()

    def x_y_to_numpy(self):
        """
        Konvertiert die geladenen Daten in numpy-Arrays für Features (X) und Labels (y).
        """
        self.df = pd.read_csv("final_zeitreihen.csv")
        self.df["category_encoded"] = self.label_encoder.fit_transform(
            self.df["Kategorie"]
        )
        y = self.df["category_encoded"].values
        x = np.array(
            self.df["Zeitreihe_padded"]
            .apply(lambda val: np.array(list(map(float, val.split(",")))))
            .tolist()
        )
        return x, y

    def classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print(report)
        print("\n")

    def confusionmatrix(self, y_test, y_pred):
        self.classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Vorhergesagte Klasse")
        plt.ylabel("Wahre Klasse")
        plt.title("Konfusionsmatrix")
        plt.show()

    def regression_statistics(self, y_test, y_pred):
        """Erstellt eine schöne Tabelle für R², MAE und RSME-Werte für die Testdaten"""
        print("\n")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        table = PrettyTable()
        table.field_names = ["R²", "MAE", "RSME"]
        table.add_row([r2, mae, rmse])
        print(table)


class loadAggData:

    def __init__(self):
        self.df = pd.DataFrame()
        self.label_encoder = LabelEncoder()
        self.columns = [
            "P_mean",
            "F_mean",
            "D_mean",
            "T_mean",
            "P_med",
            "F_med",
            "D_med",
            "T_med",
            "P_std",
            "F_std",
            "D_std",
            "T_std",
            "P_max",
            "F_max",
            "D_max",
            "T_max",
            "T_sum",
        ]
        self.Z_min = 0
        self.Z_max = 0
        self.train_indices = None
        self.test_indices = None

    def normalize(self, x):
        """Max-Min-Normalisierungsfunktion"""
        return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    def read_min_max(self):
        self.Z_min = self.df["Z_mean"].min()
        self.Z_max = self.df["Z_mean"].max()

    def re_min_max(self, x):

        return x * (self.Z_max - self.Z_min) + self.Z_min

    def return_normal_scale(self, y):

        return np.apply_along_axis(self.re_min_max, axis=0, arr=y)

    def load_and_split_classification_data(
        self,
        filter_columns=None,
        normalize=False,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    ):
        self.df = pd.read_csv("aggregated_data.csv")
        x = self.df[self.columns].values

        if filter_columns:
            x = self.filter_x_by_columns(x, filter_columns)

        if normalize:
            x = np.apply_along_axis(self.normalize, axis=0, arr=x)

        self.df["category_encoded"] = self.label_encoder.fit_transform(
            self.df["Kategorie"]
        )
        y = self.df["category_encoded"].values

        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            x,
            y,
            self.df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        return x_train, x_test, y_train, y_test

    def load_and_split_regression_data(
        self,
        filter_columns=None,
        normalize=False,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    ):
        self.df = pd.read_csv("aggregated_data.csv")
        self.read_min_max()
        x = self.df[self.columns].values

        if filter_columns:
            x = self.filter_x_by_columns(x, filter_columns)

        y = self.df["Z_mean"].values

        if normalize:
            x = np.apply_along_axis(self.normalize, axis=0, arr=x)
            y = np.apply_along_axis(self.normalize, axis=0, arr=y)

        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            x,
            y,
            self.df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        return x_train, x_test, y_train, y_test

    def filter_x_by_columns(self, x, filter_columns):
        filter_indices = [self.columns.index(col) for col in filter_columns]
        return x[:, filter_indices]

    def classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print(report)
        print("\n")

    def confusionmatrix(self, y_test, y_pred):
        self.classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Vorhergesagte Klasse")
        plt.ylabel("Wahre Klasse")
        plt.title("Konfusionsmatrix")
        plt.show()

    def regression_statistics(self, y_test, y_pred, rescale=False):
        """Erstellt eine schöne Tabelle für R², MAE und RSME-Werte für die Testdaten"""
        print("\n")

        def mean_absolute_percentage_error(y_true, y_pred):
            """MAPE-Berechnung, Nullwerte in y_true werden ignoriert. Durch die Zugfestigkeit von 0 muss entfernt werden, da sonst der MAPE viel zu hoch ist."""
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0  # Filter für Nicht-Null-Werte
            return (
                np.mean(
                    np.abs(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                )
                * 100
            )

        if rescale:
            y_pred = self.return_normal_scale(y_pred)
            y_test = self.return_normal_scale(y_test)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        table = PrettyTable()
        table.field_names = ["R²", "MAE", "RSME", "MAPE"]
        table.add_row([round(r2, 3), round(mae, 3), round(rmse, 3), round(mape, 2)])
        print(table)

    def cross_validation_statistics(self, x_train, y_train, best_model):
        cv_scores = cross_val_score(
            best_model, x_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(f"Kreuzvalidierungs-RMSE (pro Fold): {cv_rmse}")
        print(f"Durchschnittliches RMSE: {cv_rmse.mean():.4f}")

    def plot_regression_results(
        self, y_test, y_pred, rescale=False, category_column="Kategorie"
    ):

        self.label_encoder.fit(self.df[category_column])

        filtered_df = self.df.loc[self.test_indices]
        categories = filtered_df[category_column]

        if len(categories) != len(y_test):
            raise ValueError(
                "Die Länge der Kategorien stimmt nicht mit den Testdaten überein."
            )

        if rescale:
            y_pred = self.return_normal_scale(y_pred)
            y_test = self.return_normal_scale(y_test)

        unique_categories = self.label_encoder.classes_

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            y_test,
            y_pred,
            c=self.label_encoder.transform(categories),
            cmap="Blues",
            edgecolor="k",
            s=70,
        )
        colorbar = plt.colorbar(
            scatter, ticks=range(len(unique_categories)), label=category_column
        )
        colorbar.ax.set_yticklabels(unique_categories)

        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            color="black",
            linestyle="--",
            label="Ideal: y_true = y_pred",
        )
        plt.title(
            "Tatsächliche Werte vs. Vorhergesagte Werte der Zugfestigkeit nach der Kategorie"
        )
        plt.xlabel("Echte Werte der Zugfestigkeit")
        plt.ylabel("Vorhergesagte Werte der Zugfestigkeit")
        plt.legend()
        plt.grid(True)
        plt.show()


class load_stat_feature_pca:
    def __init__(self, n_components=3):
        self.df = pd.DataFrame()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.columns = [
            "P_mean",
            "F_mean",
            "D_mean",
            "T_mean",
            "P_med",
            "F_med",
            "D_med",
            "T_med",
            "P_std",
            "F_std",
            "D_std",
            "T_std",
            "P_max",
            "F_max",
            "D_max",
            "T_max",
            "T_sum",
            "P_q25",
            "F_q25",
            "D_q25",
            "T_q25",
            "P_q75",
            "F_q75",
            "D_q75",
            "T_q75",
        ]
        self.train_indices = None
        self.test_indices = None

    def load_and_split_classification_data(
        self,
        filter_columns=None,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    ):
        self.df = pd.read_csv("stat_features2.csv")
        x = self.df[self.columns].values

        if filter_columns:
            x = self.filter_x_by_columns(x, filter_columns)

        x = self.scaler.fit_transform(x)
        x = self.pca.fit_transform(x)

        self.df["category_encoded"] = self.label_encoder.fit_transform(
            self.df["Kategorie"]
        )
        y = self.df["category_encoded"].values

        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            x,
            y,
            self.df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        return x_train, x_test, y_train, y_test

    def load_and_split_regression_data(
        self,
        filter_columns=None,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    ):
        self.df = pd.read_csv("stat_features2.csv")
        x = self.df[self.columns].values

        if filter_columns:
            x = self.filter_x_by_columns(x, filter_columns)

        x = self.scaler.fit_transform(x)
        x_transformed = self.pca.fit_transform(x)

        y = self.df["Z_mean"].values.reshape(-1, 1)
        y = self.y_scaler.fit_transform(y).flatten()

        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            x_transformed,
            y,
            self.df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        return x_train, x_test, y_train, y_test

    def filter_x_by_columns(self, x, filter_columns):
        filter_indices = [self.columns.index(col) for col in filter_columns]
        return x[:, filter_indices]

    def return_normal_scale(self, y_scaled=None):
        """Wandelt transformierte Hauptkomponenten und skaliertes y zurück in den ursprünglichen Maßstab."""

        if y_scaled is not None:
            y_original = self.y_scaler.inverse_transform(
                y_scaled.reshape(-1, 1)
            ).flatten()
            return y_original

    def classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print(report)
        print("\n")

    def confusionmatrix(self, y_test, y_pred):
        self.classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Vorhergesagte Klasse")
        plt.ylabel("Wahre Klasse")
        plt.title("Konfusionsmatrix")
        plt.show()

    def regression_statistics(self, y_test, y_pred, rescale=True):
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0
            return (
                np.mean(
                    np.abs(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                )
                * 100
            )

        if rescale:
            y_test = self.return_normal_scale(y_scaled=y_test)
            y_pred = self.return_normal_scale(y_scaled=y_pred)

        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        table = PrettyTable()
        table.field_names = ["R²", "MAE", "RSME", "MAPE"]
        table.add_row([round(r2, 3), round(mae, 3), round(rmse, 3), round(mape, 2)])
        print(table)

    def cross_validation_statistics(self, x_train, y_train, best_model):
        cv_scores = cross_val_score(
            best_model, x_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(f"Kreuzvalidierungs-RMSE (pro Fold): {cv_rmse}")
        print(f"Durchschnittliches RMSE: {cv_rmse.mean():.4f}")

    def plot_regression_results(
        self, y_test, y_pred, category_column="Kategorie", rescale=True
    ):
        if rescale:
            y_test = self.return_normal_scale(y_scaled=y_test)
            y_pred = self.return_normal_scale(y_scaled=y_pred)

        self.label_encoder.fit(self.df[category_column])

        filtered_df = self.df.loc[self.test_indices]
        categories = filtered_df[category_column]

        if len(categories) != len(y_test):
            raise ValueError(
                "Die Länge der Kategorien stimmt nicht mit den Testdaten überein."
            )

        unique_categories = self.label_encoder.classes_

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            y_test,
            y_pred,
            c=self.label_encoder.transform(categories),
            cmap="Blues",
            edgecolor="k",
            s=70,
        )
        colorbar = plt.colorbar(
            scatter, ticks=range(len(unique_categories)), label=category_column
        )
        colorbar.ax.set_yticklabels(unique_categories)

        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            color="black",
            linestyle="--",
            label="Ideal: y_true = y_pred",
        )
        plt.title(
            "Tatsächliche Werte vs. Vorhergesagte Werte der Zugfestigkeit nach der Kategorie"
        )
        plt.xlabel("Echte Werte der Zugfestigkeit")
        plt.ylabel("Vorhergesagte Werte der Zugfestigkeit")
        plt.legend()
        plt.grid(True)
        plt.show()


class loadAggData_subsample:

    def __init__(self):
        self.df = pd.DataFrame()
        self.label_encoder = LabelEncoder()
        self.columns = [
            "P_mean",
            "F_mean",
            "D_mean",
            "T_mean",
            "P_med",
            "F_med",
            "D_med",
            "T_med",
            "P_std",
            "F_std",
            "D_std",
            "T_std",
            "P_max",
            "F_max",
            "D_max",
            "T_max",
            "T_sum",
        ]
        self.Z_min = 0
        self.Z_max = 0
        self.train_indices = None
        self.test_indices = None

    def normalize(self, x):
        """Max-Min-Normalisierungsfunktion"""
        return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    def read_min_max(self):
        self.Z_min = self.df["Z_mean"].min()
        self.Z_max = self.df["Z_mean"].max()

    def re_min_max(self, x):

        return x * (self.Z_max - self.Z_min) + self.Z_min

    def return_normal_scale(self, y):

        return np.apply_along_axis(self.re_min_max, axis=0, arr=y)

    def pick_classes_classifier(self, sample_size):
        self.df = (
            self.df.groupby("Kategorie", group_keys=False)
            .apply(lambda x: x.sample(n=sample_size, random_state=42))
            .reset_index(drop=True)
        )

    def load_and_split_classification_data(
        self,
        filter_columns=None,
        normalize=False,
        test_size=0.25,
        shuffle=True,
        random_state=42,
        sample_size=None,
    ):
        self.df = pd.read_csv("aggregated_data.csv")
        if sample_size != None:
            self.pick_classes_classifier(sample_size)

        x = self.df[self.columns].values

        if filter_columns:
            x = self.filter_x_by_columns(x, filter_columns)

        if normalize:
            x = np.apply_along_axis(self.normalize, axis=0, arr=x)

        self.df["category_encoded"] = self.label_encoder.fit_transform(
            self.df["Kategorie"]
        )
        y = self.df["category_encoded"].values

        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            x,
            y,
            self.df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        return x_train, x_test, y_train, y_test

    def load_and_split_regression_data(
        self,
        filter_columns=None,
        normalize=False,
        test_size=0.25,
        shuffle=True,
        random_state=42,
        sample_size=None,
    ):
        self.df = pd.read_csv("aggregated_data.csv")
        if sample_size != None:
            self.df = self.df.sample(n=sample_size, random_state=random_state)
        self.read_min_max()
        x = self.df[self.columns].values

        if filter_columns:
            x = self.filter_x_by_columns(x, filter_columns)

        y = self.df["Z_mean"].values

        if normalize:
            x = np.apply_along_axis(self.normalize, axis=0, arr=x)
            y = np.apply_along_axis(self.normalize, axis=0, arr=y)

        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            x,
            y,
            self.df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        return x_train, x_test, y_train, y_test

    def filter_x_by_columns(self, x, filter_columns):
        filter_indices = [self.columns.index(col) for col in filter_columns]
        return x[:, filter_indices]

    def classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print(report)
        print("\n")

    def confusionmatrix(self, y_test, y_pred):
        self.classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.xlabel("Vorhergesagte Klasse")
        plt.ylabel("Wahre Klasse")
        plt.title("Konfusionsmatrix")
        plt.show()

    def regression_statistics(self, y_test, y_pred, rescale=False):
        """Erstellt eine schöne Tabelle für R², MAE und RSME-Werte für die Testdaten"""
        print("\n")

        def mean_absolute_percentage_error(y_true, y_pred):
            """MAPE-Berechnung, Nullwerte in y_true werden ignoriert. Durch die Zugfestigkeit von 0 muss entfernt werden, da sonst der MAPE viel zu hoch ist."""
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0  # Filter für Nicht-Null-Werte
            return (
                np.mean(
                    np.abs(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                )
                * 100
            )

        if rescale:
            y_pred = self.return_normal_scale(y_pred)
            y_test = self.return_normal_scale(y_test)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        table = PrettyTable()
        table.field_names = ["R²", "MAE", "RSME", "MAPE"]
        table.add_row([round(r2, 3), round(mae, 3), round(rmse, 3), round(mape, 2)])
        print(table)

    def cross_validation_statistics(self, x_train, y_train, best_model):
        cv_scores = cross_val_score(
            best_model, x_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(f"Kreuzvalidierungs-RMSE (pro Fold): {cv_rmse}")
        print(f"Durchschnittliches RMSE: {cv_rmse.mean():.4f}")

    def plot_regression_results(
        self, y_test, y_pred, rescale=False, category_column="Kategorie"
    ):

        self.label_encoder.fit(self.df[category_column])

        filtered_df = self.df.loc[self.test_indices]
        categories = filtered_df[category_column]

        if len(categories) != len(y_test):
            raise ValueError(
                "Die Länge der Kategorien stimmt nicht mit den Testdaten überein."
            )

        if rescale:
            y_pred = self.return_normal_scale(y_pred)
            y_test = self.return_normal_scale(y_test)

        unique_categories = self.label_encoder.classes_

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            y_test,
            y_pred,
            c=self.label_encoder.transform(categories),
            cmap="Blues",
            edgecolor="k",
            s=70,
        )
        colorbar = plt.colorbar(
            scatter, ticks=range(len(unique_categories)), label=category_column
        )
        colorbar.ax.set_yticklabels(unique_categories)

        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            color="black",
            linestyle="--",
            label="Ideal: y_true = y_pred",
        )
        plt.title(
            "Tatsächliche Werte vs. Vorhergesagte Werte der Zugfestigkeit nach der Kategorie"
        )
        plt.xlabel("Echte Werte der Zugfestigkeit")
        plt.ylabel("Vorhergesagte Werte der Zugfestigkeit")
        plt.legend()
        plt.grid(True)
        plt.show()
