import pandas as pd

def prepare_data(data):
    """
    Aggregiert die Prozessdaten je Experiment.

    Für jedes Experiment wird die Zielvariable (Z) aus der letzten Zeile entnommen 
    und folgende Features werden aggregiert:
        - P_mean, P_max, P_IQR (Power-bezogene Merkmale)
        - F_mean, F_max, F_75 (Force-bezogene Merkmale)
        - D_std, D_50 (Distance-bezogene Merkmale)

    Parameter:
        data (DataFrame): Der vollständige Datensatz mit Spalten wie P, F, D, Z und Experiment_ID.
    
    Rückgabe:
        X (DataFrame): Aggregierte Eingabe-Features.
        y (Series): Zielvariable Z.
    """
    features = []
    targets = []
    
    for experiment_id, group in data.groupby('Experiment_ID'):
        # Zielvariable: letzter Wert der Spalte Z
        target = group.iloc[-1]['Z']
        targets.append(target)
        
        feature_row = {
            # Power-Features
            'P_mean': group['P'].mean(),
            'P_max': group['P'].max(),
            'P_IQR': group['P'].quantile(0.75) - group['P'].quantile(0.25),
            
            # Force-Features
            'F_mean': group['F'].mean(),
            'F_max': group['F'].max(),
            'F_75': group['F'].quantile(0.75),
            
            # Distance-Features
            'D_std': group['D'].std(),
            'D_50': group['D'].median()
        }
        features.append(feature_row)
        
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Z')
    return X, y

def prepare_data_classification(data):
    """
    Aggregiert die Prozessdaten für eine Klassifikationsaufgabe.
    Die Zielvariable wird dabei aus einer vorhandenen Klassifikationsspalte übernommen.

    Parameter:
        data (DataFrame): Der vollständige Datensatz mit einer Klassifikationsspalte "Kategorie".
    
    Rückgabe:
        X (DataFrame): Aggregierte Eingabe-Features.
        y (Series): Zielvariable "Kategorie" für die Klassifikation.
    """
    features = []
    targets = []
    
    for experiment_id, group in data.groupby('Experiment_ID'):
        # Zielvariable: Klassifikation aus der Spalte "Kategorie"
        target = group.iloc[-1]['Kategorie']  # Stelle sicher, dass "Kategorie" existiert
        targets.append(target)
        
        feature_row = {
            # Power-Features
            'P_mean': group['P'].mean(),
            'P_max': group['P'].max(),
            'P_IQR': group['P'].quantile(0.75) - group['P'].quantile(0.25),
            
            # Force-Features
            'F_mean': group['F'].mean(),
            'F_max': group['F'].max(),
            'F_75': group['F'].quantile(0.75),
            
            # Distance-Features
            'D_std': group['D'].std(),
            'D_50': group['D'].median()
        }
        features.append(feature_row)
        
    X = pd.DataFrame(features)
    y = pd.Series(targets, name='Kategorie')
    return X, y
