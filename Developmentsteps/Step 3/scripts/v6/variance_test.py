import pandas as pd
from sklearn.decomposition import PCA

# Daten laden (Pfad anpassen)
file_path = "../../data/complete_data.csv"
data = pd.read_csv(file_path)

# Gruppieren nach Experiment_ID und Features berechnen
grouped_data = data.groupby('Experiment_ID')

# Berechnung der Features je Experiment
features = grouped_data.apply(lambda group: pd.Series({
    'P_mean': group['P'].mean(),
    'F_max': group['F'].max(),
    'D_std': group['D'].std(),
    'P_IQR': group['P'].quantile(0.75) - group['P'].quantile(0.25),
    'F_75': group['F'].quantile(0.75),
    'D_50': group['D'].median(),
    'P_max': group['P'].max(),
    'F_mean': group['F'].mean(),
    'D_mean': group['D'].mean(),
    'D_25': group['D'].quantile(0.25),
    'D_75': group['D'].quantile(0.75),
    'D_IQR': group['D'].quantile(0.75) - group['D'].quantile(0.25)
}))

# Gruppierte Zielvariable Z berechnen (letzter Wert im Experiment)
features['Z'] = grouped_data['Z'].last().values

# Feature-Gruppen definieren
power_features = ['P_mean', 'P_max', 'P_IQR']
force_features = ['F_mean', 'F_max', 'F_75']
distance_features = ['D_mean', 'D_std', 'D_50']
dynamic_features = ['D_IQR', 'D_25', 'D_75']

# Funktion zur Berechnung der erklärten Varianz und PCA-Komponenten
def calculate_pca(features, feature_names):
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(features[feature_names])
    explained_variance = pca.explained_variance_ratio_[0]
    return pca_result, explained_variance

# Berechnung der PACs und erklärten Varianzen
pca_power, var_power = calculate_pca(features, power_features)
pca_force, var_force = calculate_pca(features, force_features)
pca_distance, var_distance = calculate_pca(features, distance_features)
pca_dynamic, var_dynamic = calculate_pca(features, dynamic_features)

# Ergebnisse ausgeben
print("Erklärte Varianzanteile:")
print(f"PAC1 (Power): {var_power:.4f}")
print(f"PAC2 (Force): {var_force:.4f}")
print(f"PAC3 (Distance): {var_distance:.4f}")
print(f"PAC4 (Dynamic): {var_dynamic:.4f}")

# PACs in das DataFrame integrieren
features['PAC1'] = pca_power.flatten()
features['PAC2'] = pca_force.flatten()
features['PAC3'] = pca_distance.flatten()
features['PAC4'] = pca_dynamic.flatten()

# Korrelationen mit der Zielvariable Z berechnen
correlation_with_Z = features[['PAC1', 'PAC2', 'PAC3', 'PAC4']].corrwith(features['Z'])

# Korrelationsergebnisse ausgeben
print("\nKorrelationen der PACs mit der Zielvariable Z:")
print(correlation_with_Z)
