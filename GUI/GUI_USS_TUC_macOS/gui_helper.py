""""""""""""""""""""""""""""""""""""
"""   HILFSFUNKTIONEN FÜR GUI    """
""""""""""""""""""""""""""""""""""""

import io, os
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import re
from scipy.stats import kurtosis, skew, mode
from sklearn import model_selection, metrics, manifold
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, TransformerMixin

# Statistische Kennzahlen für die Modellerstellung
STATS = ['p_max', 'p_q75', 'p_median', 'p_f_ratio', 'p_q25', 'p_d_ratio', 'p_t_ratio', 'f_t_ratio', 'f_d_ratio', 't_max', 'p_std', 'd_std']
# Farben und Labels für Diagramme definieren
colors = ['green', 'orange', 'black', 'red', 'blue']
labels = ['OK-Schweißung', 'NEAR-OK-Sonotrodenwechsel', 'Öl auf Terminalversatz', 
          'Leitungsversatz', 'Terminalversatz']
DATA = ['ok', 'near_ok_sonowechsel', 'fehler_oel', 'fehler_leitungsversatz', 'fehler_terminversatz']
label  = dict() 
for idx, fname in enumerate(DATA):
    label[fname] = idx

class StatFeatures(BaseEstimator, TransformerMixin):
    '''
    Umwandeln der Zeitreihe in statistische Features.
    Berechnet nur spezifische Spalten basierend auf den vom Nutzer angegebenen Parametern.
    '''
    def __init__(self, columns=STATS):
        '''
        :param columns: Namen der gewünschten statistischen Größen ("p_mean", "f_q25", "d_median", "ti").
        '''
        self.columns = columns if columns else []
        self.column_names = []

    def fit(self, X=None, y=None):
        return self

    def transform(self, keys, kurven):
        '''
        Umwandeln der Schweißkurven in ein Array[versuche, features].
        :param keys: Liste der Schlüssel für die Schweißkurven.
        :param kurven: Dictionary der Schweißkurven.
        :return: NumPy-Array mit extrahierten Features.
        '''
        features = []

        for key in keys:
            kurve = kurven[key]
            feature = []
            epsilon = 1e-6  # Kleiner Wert zur Vermeidung von Division durch Null

            # Statistiken basierend auf den ausgewählten Spalten berechnen
            for col in self.columns:
                # ---- POWER ---- #
                if col == "p_mean":
                    feature.append(kurve.power.mean())
                elif col == "p_median":
                    feature.append(kurve.power.median())
                elif col == "p_std":
                    feature.append(kurve.power.std())
                elif col == "p_max":
                    feature.append(kurve.power.max())
                elif col.startswith("p_q"):
                    q = float(col.split("q")[1]) / 100
                    feature.append(np.quantile(kurve.power, q))
                elif col == "p_iqr_range":
                    q75, q25 = np.percentile(kurve.power, [75, 25])
                    feature.append((q75 - q25)/(kurve.power.max()-kurve.power.min()))
                elif col == "p_f_ratio":
                    sum_power = kurve.power.sum()
                    sum_force = kurve.force.sum()
                    feature.append(sum_power / (sum_force + epsilon))
                elif col == "p_d_ratio":
                    sum_power = kurve.power.sum()
                    sum_dist = kurve.dist.sum()
                    feature.append(sum_power / (sum_dist + epsilon))
                elif col == "p_t_ratio":
                    sum_power = kurve.power.sum()
                    sum_time = kurve.ms.sum()
                    feature.append(sum_power / (sum_time + epsilon))
                #elif col == "p_b75":
                #    p_max = kurve.power.max()
                #    p_b75 = kurve.loc[kurve.power >= 0.75*p_max].iloc[0].ms
                #    feature.append(p_b75)
                # ---- FORCE ---- #
                elif col == "f_mean":
                    feature.append(kurve.force.mean())
                elif col == "f_median":
                    feature.append(kurve.force.median())
                elif col == "f_std":
                    feature.append(kurve.force.std())
                elif col == "f_max":
                    feature.append(kurve.force.max())
                elif col.startswith("f_q"): 
                    q = float(col.split("q")[1]) / 100
                    feature.append(np.quantile(kurve.force, q))
                elif col == "f_iqr_range":
                    q75, q25 = np.percentile(kurve.force, [75, 25])
                    feature.append((q75 - q25)/(kurve.force.max()-kurve.force.min()))
                elif col == "f_d_ratio":
                    sum_dist = kurve.force.sum()
                    sum_time = kurve.dist.sum()
                    feature.append(sum_dist / (sum_time + epsilon))
                elif col == "f_t_ratio":
                    sum_dist = kurve.force.sum()
                    sum_time = kurve.ms.sum()
                    feature.append(sum_dist / (sum_time + epsilon))
                #elif col == "f_b75":
                #    f_max = kurve.force.max()
                #    f_b75 = kurve.loc[kurve.force >= 0.75*f_max].iloc[0].ms
                #    feature.append(f_b75)
                # ---- DIST ---- #
                elif col == "d_mean":
                    feature.append(kurve.dist.mean())
                elif col == "d_median":
                    feature.append(kurve.dist.median())
                elif col == "d_std":
                    feature.append(kurve.dist.std())
                elif col == "d_max":
                    feature.append(kurve.dist.max())
                elif col.startswith("d_q"):
                    q = float(col.split("q")[1]) / 100
                    feature.append(np.quantile(kurve.dist, q))
                elif col == "d_iqr_range":
                    q75, q25 = np.percentile(kurve.dist, [75, 25])
                    feature.append((q75 - q25)/(kurve.dist.max()-kurve.dist.min()))
                elif col == "d_t_ratio":
                    sum_dist = kurve.dist.sum()
                    sum_time = kurve.ms.sum()
                    feature.append(sum_dist / (sum_time + epsilon))
                elif col == "d_b75":
                    d_max = kurve.dist.max()
                    d_b75 = kurve.loc[kurve.dist >= 0.75*d_max].iloc[0].ms
                    feature.append(d_b75)
                # ---- ZEITMERKMALE ---- #
                elif col == "t_max":
                    feature.append(kurve.ms.max())   
                else: 
                    return print(f"Spalte", col, "nicht vorhanden!")
            features.append(np.array(feature))

        return np.stack(features)
    
def plot_pred(y_data, festigkeit, label_data, titel, perfect, SCALE):
    """
    Plottet ein Streudiagramm für die Vorhersage der Zugfestigkeit
    """
    y_data = np.array(y_data)
    festigkeit = np.array(festigkeit)
    label_data = np.array(label_data)

    fig, ax = plt.subplots(figsize=(4, 3))
    if label_data is not None and label_data.any():
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = label_data == i
            plt.scatter(
                festigkeit[mask]*SCALE, 
                y_data[mask]*SCALE, 
                c=color, 
                label=label, 
                alpha=0.7, 
                edgecolor='k', 
                s=50
            )
    else:
        plt.scatter(
            festigkeit*SCALE, 
            y_data*SCALE, 
            #label=label, 
            alpha=0.7, 
            edgecolor='k', 
            s=50
        )
    if perfect == 'mit Linie':
        plt.xlim(0, 3000)
        plt.ylim(0, 3000)
        plt.plot([0, 3500], [0, 3500], "k:", label="Perfekte Vorhersage")
    plt.title(titel, fontsize=12, font='Arial', weight='bold', pad=10)
    plt.xlabel('Realität (Zugfestigkeit in [MPa])', fontsize=10)
    plt.ylabel('Vorhersage (Zugfestigkeit in [MPa])', fontsize=10)
    plt.legend(fontsize=9, loc='best', framealpha=0.8)
    plt.tight_layout()
    return fig


def plotIQRcurve(kurven_df, feature):
    """ plot feature with IQR for different labels """ 

    df = kurven_df[['key', 'label_name', 'ms', feature]]
    df_grp_med = df.groupby(['label_name', 'ms']).agg({ feature: ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]}).reset_index()
    df_grp_med.columns = ['label_name', 'ms', 'p_med', 'quantil_25', 'quantil_75']

    #plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(6, 3))

    # Für jedes Label eine Linie mit Schattierung zeichnen
    for label, color in zip(labels, colors):
        df_label = df_grp_med[df_grp_med['label_name'] == label]
        
        # Median-Linie
        plt.plot(df_label['ms'], df_label['p_med'], color=color, label=label, linewidth=1.0)

        # IQR-Bereich als Schattierung
        plt.fill_between(df_label['ms'], df_label['quantil_25'], df_label['quantil_75'], 
                        color=color, alpha=0.2)  # alpha=0.2 für Transparenz
    
    feature_txt = {
        'power': 'Energie in [W]',
        'force': 'Pressenkraft in [N]',
        'dist':  'Sonotrodenvorschub in [mm]'
    }.get(feature, ' ')
    plt.title(f"{feature_txt.split()[0]} im Zeitverlauf mit IQR-Bereich für verschiedene Klassen", fontsize=10, weight='bold', y=1.02)
    plt.xlabel("Zeit in [ms]", fontsize=9)
    plt.xticks(np.arange(0,2750,250))
    plt.ylabel(feature_txt, fontsize=9)
    #plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def get_kurven_df(kurven):
    # Dictionairy Kurven in DataFrame umwandeln
    combined_data = []
    for key, df in kurven.items():
        df['key'] = key
        combined_data.append(df)
    kurven_df = pd.concat(combined_data, ignore_index=True)
    columns_order = ['key'] + [col for col in kurven_df.columns if col != 'key']
    kurven_df = kurven_df[columns_order]
    kurven_df = kurven_df.set_index(['key'])
    return kurven_df


def class_curves(versuche, kurven, feature):
    """
    Plottet Schweißkurven in Abhängigkeit von Klassifikationen.

    Parameter:
        versuche (DataFrame): DataFrame mit den Spalten 'key' (Versuchsnummer) und 'label' (Klassifikation)
        kurven (dict): Schweißkurven mit Schlüssel 'key', Wert ist DataFrame mit den Spalten 'ms' und dem angegebenen Feature
        feature (str): darzustellendes Feature, 'power', 'force' oder 'dist'

    Die Funktion erzeugt einen Plot der Schweißkurven mit farblich unterschiedenen Klassifikationen.
    """

    kurven = {int(k): v for k, v in kurven.items()}
    #versuche['key'] = versuche['key'].astype(int)
    #kurven['key'] = kurven['key'].astype(int)

    fig, ax = plt.subplots(figsize=(6, 4))

    for _, v in versuche.iterrows():
        kurve = kurven[v['key']]
        c = int(v['klasse'])
        plt.plot(kurve.ms, kurve[feature], c=colors[c], linewidth=0.5, alpha=0.5)

    handles = [mlines.Line2D([0], [0], color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=handles, loc='lower right', framealpha=0.7)

    feature_txt = {
        'power': 'Energie in [W]',
        'force': 'Pressenkraft in [N]',
        'dist':  'Sonotrodenvorschub in [mm]'
    }.get(feature, ' ')
    plt.title(f"Darstellung der Schweißkurven in Abhängigkeit der {feature_txt.split()[0]}", weight='bold', fontsize=10)
    plt.xlabel("Zeit in [ms]", fontsize=9)
    plt.ylabel(feature_txt)
    fig.tight_layout()
    return fig


def zug_curves(versuche, kurven, feature):
    fig, ax = plt.subplots()
    max_fest = versuche['festigkeit'].max()
    cmap = matplotlib.colormaps['rainbow']
    norm = matplotlib.colors.Normalize(0, max_fest)
    for _, v in versuche.iterrows():
        kurve = kurven[v['nr']]
        #m = v['label']
        color = cmap(v['festigkeit']/max_fest)
        plt.plot(kurve.ms, kurve[feature], c=color, alpha=0.5, linewidth=0.5)
    # Farbskala
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), orientation='vertical', ax=ax, label='Zugfestigkeit in [MPa]')
    # Titel und Achsenbeschriftung
    feature_txt = {
        'power': 'Energie in [W]',
        'force': 'Pressenkraft in [N]',
        'dist':  'Sonotrodenvorschub in [mm]'
    }.get(feature, ' ')
    plt.title(f"Darstellung der Schweißkurven in Abhängigkeit \n von {feature_txt.split()[0]} und Zugfestigkeit", weight='bold', fontsize=14)
    plt.xlabel("Zeit in [ms]")
    plt.ylabel(feature_txt)
    fig.tight_layout()