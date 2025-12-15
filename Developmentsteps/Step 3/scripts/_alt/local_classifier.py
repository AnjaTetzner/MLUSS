from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def prepare_classification_data(data):
    """
    Bereitet die aggregierten Daten für die Klassifikation vor.
    
    Parameter:
        data (DataFrame): Der Datensatz mit P, F, D, Kategorie und Experiment_ID.
    
    Rückgabe:
        X (DataFrame): Aggregierte Eingabe-Features.
        y (Series): Zielvariable (Kategorie).
    """
    # Ergebnisse speichern
    features = []
    targets = []
    
    # Gruppieren nach Experiment_ID
    for experiment_id, group in data.groupby('Experiment_ID'):
        # Zielvariable "Kategorie" aus der letzten Zeile des Prozesses extrahieren
        target = group.iloc[-1]['Kategorie']
        targets.append(target)
        
        # Aggregation der Eingabefeatures
        feature_row = {
            'Experiment_ID': experiment_id,
            'P_mean': group['P'].mean(),
            'P_max': group['P'].max(),
            'F_mean': group['F'].mean(),
            'F_max': group['F'].max(),
            'D_mean': group['D'].mean(),
            'D_max': group['D'].max(),
        }
        features.append(feature_row)
    
    # Features und Zielvariable in DataFrames umwandeln
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Kategorie')
    
    return X, y

def classify_data(X, y):
    """
    Führt eine Klassifikation mit einem Random Forest durch und gibt die Ergebnisse aus.
    
    Parameter:
        X (DataFrame): Eingabe-Features.
        y (Series): Zielvariable (Kategorie).
    
    Rückgabe:
        model: Das trainierte Random Forest Modell.
    """
    # Training und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
    
    # Random Forest Klassifikator trainieren
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Vorhersagen auf Testdaten
    y_pred = model.predict(X_test)
    
    # Klassifikationsbericht und Genauigkeit ausgeben
    print("Klassifikationsbericht:")
    print(classification_report(y_test, y_pred))
    print(f"Genauigkeit: {accuracy_score(y_test, y_pred):.4f}")
    
    return model

## Definiere eine Klasse, um die Daten zu verwalten
#class DataStorage:
#    def __init__(self, file_path):
#        self.file_path = file_path
#        self.data = None
#
#    def load_data(self):
#        """Lädt die Daten aus der CSV-Datei."""
#        self.data = pd.read_csv(self.file_path)
#
#    def get_data(self):
#        """Gibt die geladenen Daten zurück."""
#        if self.data is None:
#            raise ValueError("Daten wurden noch nicht geladen. Verwende 'load_data', um sie zu laden.")
#        return self.data
#        
#if __name__ == "__main__":
#    file_path = "complete_data.csv"
#    data_storage = DataStorage(file_path)
#
#    # Daten laden
#    data_storage.load_data()
#
#    # Geladene Daten anzeigen
#    data = data_storage.get_data()
#
#    # Testcode
#    print("Das Modul funktioniert korrekt!")
#
#        # Daten vorbereiten
#    X, y = prepare_classification_data(data)
#    
#    # Klassifikationsmodell trainieren
#    rf_model = classify_data(X, y)