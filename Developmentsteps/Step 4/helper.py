""""""""""""""""""""""""
""" HILFSFUNKTIONEN  """
""""""""""""""""""""""""

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
import tsgm

BASE_DIR = 'Rohdaten_renamed'
DATA = ['ok', 'near_ok_sonowechsel', 'fehler_oel', 'fehler_leitungsversatz', 'fehler_terminversatz']

label  = dict() 
for idx, fname in enumerate(DATA):
    label[fname] = idx

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def read_special_csv(path):
    """
    Einlesen der Maschinendaten.
    Vorspann muss entfernt werden, CSV beginnt nach Zeile #CSV-START#
    """
    res = list()
    flag = False
    for line in open(path):
        if line.strip() == '#CSV-START#':
            flag = True
            continue
        if flag:
            res.append(line)
    stream = io.StringIO('\n'.join(res))
    df = pd.read_csv(stream, sep=';')
    df.columns = ['ms', 'power', 'force', 'dist']
    return df

def read_kurven_gg(data_dir, kurven_dict, prefix):
    '''
    Einlesen aller Schweisskurven

    data_dir: Verzeichnis in der die Schweisskurven liegen
    kurven_dict: Kurve wird als DataFrame im Dictionary gespeichert
    {prefix}_{nr}: Key
    '''

    max_row_count = 0
    min_row_count = 999999999
    count = 0
    for fname in os.listdir(data_dir):
        base, ext = os.path.splitext(fname)
        if ext != ".csv":
            continue
        count += 1
        df = read_special_csv(os.path.join(data_dir, fname))
        kurven_dict[prefix+base] = df
        if df.shape[0] > max_row_count:
            max_row_count = df.shape[0]
        if df.shape[0] < min_row_count:
            min_row_count = df.shape[0]
    print(f'{prefix}: count {count} rows {min_row_count}...{max_row_count}')


def read_kurven(data_dir, df_zug, kurven_dict, prefix):
    """
    Einlesen aller Schweisskurven

    data_dir: Verzeichnis in der die Schweisskurven liegen
    kurven_dict: Kurve wird als DataFrame im Dictionary gespeichert
    {prefix}_{nr}: Key
    """
    max_row_count = 0
    min_row_count = 999999999
    for nr in df_zug.nr:
        df = read_special_csv(os.path.join(data_dir, 'schweisskurven', nr+'.csv'))
        kurven_dict[prefix+nr] = df
        if df.shape[0] > max_row_count:
            max_row_count = df.shape[0]
        if df.shape[0] < min_row_count:
            min_row_count = df.shape[0]
    # print(f"{df_zug['group'].loc[0].ljust(22)} -> {len(df_zug)} rows {min_row_count}...{max_row_count}")

def read_data_orig():
    kurven = dict()
    zugversuche = pd.DataFrame()

    for data in DATA:
        zugversuch = pd.read_csv(os.path.join(BASE_DIR, data, 'zugversuch.csv'), sep=';', decimal=',')
        zugversuch.columns = ['nr', 'festigkeit']
        prefix = data + '_'
        zugversuch['nr']  = zugversuch.nr.astype('str')
        zugversuch['key'] = prefix + zugversuch.nr
        # zugversuch['group'] = data
        zugversuch['label'] = label[data]
        zugversuch['label_name'] = zugversuch['label'].map({0: 'OK-Schweißung', 
                                                            1: 'NEAR-OK-Sonotrodenwechsel', 
                                                            2: 'Öl auf Terminalversatz', 
                                                            3: 'Leitungsversatz', 
                                                            4: 'Terminalversatz'})
        zugversuche = pd.concat([zugversuche, zugversuch])
        # gleichzeitig Schweisskurven einlesen in -> kurven
        read_kurven(os.path.join(BASE_DIR, data), zugversuch, kurven, prefix)
    zugversuche = zugversuche[['nr', 'key', 'label', 'label_name', 'festigkeit']]
    return zugversuche, kurven

def read_data():
    """ read data files """
    zugversuche = pd.read_csv("Rohdaten_renamed/zugversuche.csv", sep=';', decimal=',')
    kurven_df   = pd.read_csv("Rohdaten_renamed/kurven.csv", sep=';', decimal=',')
    kurven = {
        key: group.drop(columns=['key']).reset_index(drop=True)
        for key, group in kurven_df.groupby('key')
    }
    versuche_train = pd.read_csv("Rohdaten_renamed/versuche_train.csv", sep=';', decimal=',')
    versuche_test  = pd.read_csv("Rohdaten_renamed/versuche_test.csv", sep=';', decimal=',')
    return zugversuche, kurven, versuche_train, versuche_test


def get_kurven_df(kurven):
    """ transform kurven dictionairy to dataframe """
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
    
def save_train_test(X=None):
    """ get trainings- and testdata set 
    
    X: Anzahl der Datensätze pro Klasse
    """
    zugversuche, kurven = read_data_orig()
    zugversuche = zugversuche.set_index(['key'])
    zugversuche.to_csv("Rohdaten_renamed/zugversuche.csv", sep=';', decimal=',')
    kurven_df = get_kurven_df(kurven)
    kurven_df.to_csv("Rohdaten_renamed/kurven.csv", sep=';', decimal=',')

    # gleiche Aufteilung der Klassen 
    """
    zugversuche_split = []
    for label in zugversuche['label'].unique():
        label_data = zugversuche[zugversuche['label'] == label]
        
        if label == 2:  # Klasse Öl soll mindestens 20% 0-Versuche enthalten
            min_zero_count = max(1, int(0.2 * X))  # mindestens 1, falls 20% von X < 1
            zero_data = label_data[label_data['festigkeit'] == 0]
            positive_data = label_data[label_data['festigkeit'] > 0]
            # mindestens 20% mit Zugfestigkeit = 0 
            zero_sample = zero_data.sample(n=min(min_zero_count, len(zero_data)), random_state=42)
            # Restliche Datensätze mit Zugfestigkeit > 0 auffüllen
            available_zero_count = min(len(zero_data), max(min_zero_count, X - len(positive_data)))
            available_positive_count = min(len(positive_data), X - available_zero_count)
            
            # Ziehe verfügbare Datensätze
            zero_sample = zero_data.sample(n=available_zero_count, random_state=42, replace=False)
            positive_sample = positive_data.sample(n=available_positive_count, random_state=42, replace=False)
            
            # Kombiniere die ausgewählten Datensätze
            label_data_sample = pd.concat([positive_sample, zero_sample]) 
        else:
            # Für andere Klassen einfach X Datensätze zufällig ziehen
            label_data_sample = label_data.sample(n=min(len(label_data), X), random_state=42)
        
        zugversuche_split.append(label_data_sample)
    zugversuche_split = pd.concat(zugversuche_split)
    """
    # Daten mischen & in Trainings- und Testdaten aufteilen
    versuche_train, versuche_test = model_selection.train_test_split(zugversuche, train_size=0.8, test_size=0.2, random_state=42) #, stratify=zugversuche_split['label'])
    versuche_train.to_csv("Rohdaten_renamed/versuche_train.csv", sep=';', decimal=',')
    versuche_test.to_csv( "Rohdaten_renamed/versuche_test.csv",  sep=';', decimal=',')
    return "files saved (zugversuche, kurven, versuche_train, versuche_test)"


class Subsample(BaseEstimator, TransformerMixin):
    """
    Reduktion (Subsampling) von Zeitreihendaten aus Schweisskurven 
    -> 2 Modi: (feste ts_len oder dynamische Intervalle)
       = entweder auf fixe Länge kürzen oder gesamte Länge in subsample-gleichlange Intervalle unterteilen

    ts_len: maximale Länge (ts_len=None wenn gesetzt, wird Zeitreihe auf fixe Länge gekürzt)
    subsample: Rate, nur jeder subsample-te Punkt der gekürzten Zeitreihe wird ausgewählt

    Ergebnis: aus mehreren Schweisskurven (DataFrames) einen 3D-Tensor mit Dimensionen (versuche, zeitpunkte, features) erzeugen
    """
    def __init__(self, subsample, ts_len=None):
        self.subsample = subsample
        self.ts_len = ts_len

    def fit(self):
        return self

    def subsample_array(self, array):
        """
        teilt die Zeitreihe basierend auf dem Modus auf
        """
        if self.ts_len:
            array = array[:self.ts_len, :]  # kürzt array auf ts_len
            # step = self.subsample         # Schrittweite entspricht der Subsampling-Rate
            # step = self.ts_len // self.subsample
        n = array.shape[0]
        step = n // self.subsample                          # Dynamische Schrittweite basierend auf der Gesamtlänge n
        return array[:step * self.subsample:step, :].copy() # Wählt Punkte mit der berechneten Schrittweite aus

    def transform(self, keys, kurven):
        """
        erzeugt 3D tensor [versuch, zeitpunkt, feature]
        """
        st = [
            np.expand_dims(self.subsample_array(kurven[key].values[:, 1:]), axis=0)
            for key in keys
        ]
        return np.vstack(st)


class TsMaxScaler(BaseEstimator, TransformerMixin): 
    """
    MinMax-Scaler für jedes Feature über alle Zeitreihen/Schweißkurven/Versuche
    """
    def __init__(self):
        self.max = None       # Definition Variable
    def fit(self, X, y=None): # berechnet für jedes feature max, behält 3D array
        self.max = X.max(axis=(0,1), keepdims=True)
        return self
    def transform(self, X):   # teilt jeden Wert durch max
        return (X/self.max).copy()
    

class StatFeatures(BaseEstimator, TransformerMixin):
    '''
    Umwandeln der Zeitreihe in statistische Features.
    Berechnet nur spezifische Spalten basierend auf den vom Nutzer angegebenen Parametern.
    '''
    def __init__(self, columns=None):
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


def augment_curves(X, y, n_gauss=5, sd_gauss=0.02, n_magnitude=5, sigma=0.02, sd_y=0.05):
    '''
    Augmentieren mit Gauß-Rauschen und Magnitude warping
    '''
    model_gauss = tsgm.models.augmentations.GaussianNoise()
    model_mag = tsgm.models.augmentations.MagnitudeWarping()
    samples = list()
    targets = list()
    for i in range(X.shape[0]):
        # add original value
        samples.append(X[i:i+1,:,:])
        targets.append(y[i])
        val_max = np.max(X[i,:,:], axis=0, keepdims=True)
        var_gauss = (sd_gauss*val_max)**2
        aug_gauss = model_gauss.generate(X=X[i:i+1,:,:], n_samples=n_gauss, variance=var_gauss)
        samples.append(aug_gauss)
        for k in range(n_gauss):
            y_aug_gauss = y[i]*(1+np.random.normal(loc=0, scale=sd_y))
            targets.append(y_aug_gauss)
        aug_mag = model_mag.generate(X=X[i:i+1,:,:], n_samples=n_magnitude, sigma=sigma)
        samples.append(aug_mag)
        for k in range(n_magnitude):
            y_aug_mag = y[i]*(1+np.random.normal(loc=0, scale=sd_y))
            targets.append(y_aug_mag)
    return np.vstack(samples), np.array(targets)




""""""""""""""""""""""""""""""""""""""""""
""" HILFSFUNKTIONEN FÜR DARSTELLUNGEN  """
""""""""""""""""""""""""""""""""""""""""""

# Farben und Labels für Diagramme definieren
colors = ['green', 'orange', 'black', 'red', 'blue']
labels = ['OK-Schweißung', 'NEAR-OK-Sonotrodenwechsel', 'Öl auf Terminalversatz', 
          'Leitungsversatz', 'Terminalversatz']

def class_curves(versuche, kurven, feature):
    """
    Plottet Schweißkurven in Abhängigkeit von Klassifikationen.

    Parameter:
        versuche (DataFrame): DataFrame mit den Spalten 'key' (Versuchsnummer) und 'label' (Klassifikation)
        kurven (dict): Schweißkurven mit Schlüssel 'key', Wert ist DataFrame mit den Spalten 'ms' und dem angegebenen Feature
        feature (str): darzustellendes Feature, 'power', 'force' oder 'dist'

    Die Funktion erzeugt einen Plot der Schweißkurven mit farblich unterschiedenen Klassifikationen.
    """
    for _, v in versuche.iterrows():
        kurve = kurven[v['key']]
        c = v['label']
        plt.plot(kurve.ms, kurve[feature], c=colors[c], linewidth=0.5, alpha=0.5)

    handles = [mlines.Line2D([0], [0], color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=handles, loc='lower right', framealpha=0.7)

    feature_txt = {
        'power': 'Energie in [W]',
        'force': 'Pressenkraft in [N]',
        'dist':  'Sonotrodenvorschub in [mm]'
    }.get(feature, ' ')
    plt.title(f"Darstellung der Schweißkurven\nin Abhängigkeit der {feature_txt.split()[0]}", weight='bold', fontsize=14)
    plt.xlabel("Zeit in [ms]")
    plt.ylabel(feature_txt)
    #plt.savefig(feature_txt.split()[0]+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def zug_curves(versuche, kurven, feature):
    fig, ax = plt.subplots()
    max_fest = versuche['festigkeit'].max()
    cmap = matplotlib.colormaps['rainbow']
    norm = matplotlib.colors.Normalize(0, max_fest)
    for _, v in versuche.iterrows():
        kurve = kurven[v['key']]
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
    # Plot speichern oder anzeigen
    #plt.savefig(feature_txt.split()[0]+'.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_curves(X, ax, y, feature):
    """
    Plottet ein Liniendiagramm

    X: Die Eingabedaten (Tensor oder Array)
    ax: Der Index der Achse (für die Auswahl einer bestimmten Dimension)
    y: Die Labels, die die Klasse der Kurven bestimmen
    feature: Das Feld, nach dem die Kurve kategorisiert wird ('power', 'force', 'dist')
    """
    # Zeitschritte (x-Achse)
    ti = np.arange(0, X.shape[1]) 
    # Schleife über alle Kurven (Datenreihen)
    for i in range(X.shape[0]):
        plt.plot(ti, X[i,:,ax], c=colors[y[i]], linewidth=0.5, alpha=0.5)
    handles = [mlines.Line2D([0], [0], color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=handles, loc='lower right', framealpha=0.7)
    feature_txt = {
        'power': 'Energie in [W]',
        'force': 'Pressenkraft in [N]',
        'dist':  'Sonotrodenvorschub in [mm]'
    }.get(feature, ' ')
    plt.title(f"Darstellung der Schweißkurven in Abhängigkeit der {feature_txt.split()[0]}", weight='bold', fontsize=14)
    plt.xlabel("Zeit in [ms]")
    plt.ylabel(feature_txt)
    #plt.savefig(feature_txt.split()[0]+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_zug_curves(X, axis, y, field):
    fig, ax = plt.subplots()
    max_y = np.max(y)
    cmap = matplotlib.colormaps['rainbow']
    norm = matplotlib.colors.Normalize(0, max_y)
    ti = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        color = cmap(y[i]/max_y)
        plt.plot(ti, X[i,:,axis], c=color, alpha=0.7, linewidth=0.5)
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), orientation='vertical', ax=ax, label='Zugfestigkeit in [MPa]')
    feature_txt = {
        'power': 'Energie in [W]',
        'force': 'Pressenkraft in [N]',
        'dist':  'Sonotrodenvorschub in [mm]'
    }.get(field, ' ')
    plt.title(f"Darstellung der Schweißkurven in Abhängigkeit \n von {feature_txt.split()[0]} und Zugfestigkeit", weight='bold', fontsize=14)
    plt.xlabel("Zeit in [ms]")
    plt.ylabel(feature_txt)
    #plt.savefig(feature_txt.split()[0]+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion(y_true, y_pred, txt):
    """
    Plottet eine Konfusionsmatrix
    
    Parameter:
    - y_true: Wahre Klassen
    - y_pred: Vorhergesagte Klassen
    """
    conf = metrics.confusion_matrix(y_true, y_pred)
    #plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt='d', cbar_kws={'label': 'Anzahl'}, cmap="rocket_r")
    plt.title('Konfusionsmatrix für '+txt, weight='bold')
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Wahre Klasse')
    # Legende
    legend_text = "\n".join([f"{i} → {label}" for i, label in enumerate(labels)])
    plt.gcf().text(0.9, 0.24, legend_text, fontsize=11, va='top', ha='left') 
    plt.show()
    # plt.savefig('confusion_matrix.png')


def plot_pred(y_data, festigkeit, label_data, titel, perfect, SCALE):
    """
    Plottet ein Streudiagramm für die Vorhersage der Zugfestigkeit
    """
    #plt.figure(figsize=(10, 6))
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
    plt.title(titel, fontsize=14, weight='bold')
    plt.xlabel('Realität (Zugfestigkeit in [MPa])')
    plt.ylabel('Vorhersage (Zugfestigkeit in [MPa])')
    plt.legend(fontsize=9, loc='best', framealpha=0.8)
    #plt.tight_layout()
    #plt.savefig('svr_train.png', dpi=300)
    plt.show()


def get_metric_table_regr(festigkeit_train, y_pred_train, festigkeit_test, y_pred_test, SCALE):
    """ 
    Gibt Metriken für Regressionsmodelle als Dataframe zurück 

    Parameter:
    - festigkeit_train: Wahre Werte (Trainingsdaten)
    - festigkeit_test:  Wahre Werte (Testdaten)
    - y_pred_train: Vorhergesagte Werte (Trainingsdaten)
    - y_pred_test:  Vorhergesagte Werte (Testdaten)
    
    Returns:
    - results: DataFrame mit den berechneten Metriken
    """
    festigkeit_train = festigkeit_train*SCALE
    festigkeit_test  = festigkeit_test*SCALE
    y_pred_train = y_pred_train*SCALE
    y_pred_test  = y_pred_test*SCALE

    mse_train = metrics.mean_squared_error(festigkeit_train, y_pred_train)
    mse_test  = metrics.mean_squared_error(festigkeit_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test  = np.sqrt(mse_test)
    mae_train = metrics.mean_absolute_error(festigkeit_train, y_pred_train)
    mae_test  = metrics.mean_absolute_error(festigkeit_test, y_pred_test)
    r2_train  = metrics.r2_score(festigkeit_train, y_pred_train)
    r2_test   = metrics.r2_score(festigkeit_test, y_pred_test)

    results = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R²"],
        "Train":  [mse_train, rmse_train, mae_train, r2_train],
        "Test":   [mse_test,  rmse_test,  mae_test,  r2_test]
    }).set_index(["Metric"])
    # Formatierung basierend auf der Metrik
    results["Train"] = results.index.map(lambda metric: f"{results.loc[metric, 'Train']:.3f}" if metric == "R2" else f"{results.loc[metric, 'Train']:.2f}")
    results["Test"]  = results.index.map(lambda metric: f"{results.loc[metric, 'Test']:.3f}"  if metric == "R2" else f"{results.loc[metric, 'Test']:.2f}")

    return results


def get_metric_table_class(label_train, pred_train, label_test, pred_test):
    """ 
    Gibt Metriken für Klassifikationsmodelle als Dataframe zurück 

    Parameter:
    - label_train: Klassen der Trainingsdaten
    - label_test:  Klassen der Testdaten
    - pred_train:  Vorhergesagte Klassen (Trainingsdaten)
    - pred_test:   Vorhergesagte Klassen (Testdaten)
    
    Returns:
    - results: DataFrame mit den berechneten Metriken
    """
    acc_train = metrics.accuracy_score(label_train, pred_train)
    acc_test  = metrics.accuracy_score(label_test,  pred_test)
    precision_train = metrics.precision_score(label_train, pred_train, average='weighted')
    precision_test  = metrics.precision_score(label_test,  pred_test,  average='weighted')
    recall_train = metrics.recall_score(label_train, pred_train, average='weighted')
    recall_test  = metrics.recall_score(label_test,  pred_test,  average='weighted')
    f1_train = metrics.f1_score(label_train, pred_train, average='weighted')
    f1_test  = metrics.f1_score(label_test,  pred_test,  average='weighted')
    #loss_train = metrics.log_loss(y_true=label_train, y_pred=pred_train)
    #loss_test  = metrics.log_loss(y_true=label_test,  y_pred=pred_test)

    results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Train":  [acc_train, precision_train, recall_train, f1_train],
        "Test":   [acc_test,  precision_test,  recall_test,  f1_test]
    }).set_index(["Metric"])
    results["Train"] = results["Train"].apply(lambda x: f"{x:.3f}")
    results["Test"]  = results["Test"].apply(lambda x: f"{x:.3f}")

    print(classification_report(label_test, pred_test, digits=3, target_names=DATA))

    return results


def plot_roc_curve(label_train, label_test, y_score):
    """
    Plottet eine ROC-Kurve
    """
    # Binarisierung der Labels für Multi-Klassen-ROC (falls mehrere Klassen vorliegen)
    classes = sorted(set(label_train))  
    label_test_binarized = label_binarize(label_test, classes=classes)

    # Berechnung der ROC-Kurve und AUC für jede Klasse
    fpr     = dict()  # False Positive Rate
    tpr     = dict()  # True Positive Rate
    roc_auc = dict()  # AUC-Werte

    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(label_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Plotten der ROC-Kurve
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
                label=f'{labels[cls]} (AUC = {roc_auc[i]:.2f})')

    # plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Kurve', weight='bold')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def plotTSNE(X_train_scaled, festigkeit_train, label_train, feature, graph):
    """ plot t-SNE
    
    feature: 'Energie', 'Kraft', 'Distanz', 'all'
    graph: to plot 'Klassen' or 'Zugfestigkeit' 
    """
    feature_num = {
        'Energie': 0,
        'Kraft': 1,
        'Distanz':  2
    }.get(feature, ' ')

    if feature == "all":
        X_train_projected = manifold.TSNE(n_components=2, learning_rate='auto', random_state=42, init='pca').fit_transform(X_train_scaled)
    else:
        X_train_projected = manifold.TSNE(n_components=2, learning_rate='auto', random_state=42, init='pca').fit_transform(X_train_scaled[:,:,feature_num])

    if graph == "Zugfestigkeit":
        scatter = plt.scatter(
            X_train_projected[:, 0], 
            X_train_projected[:, 1], 
            c=festigkeit_train, 
            cmap='rainbow', 
            alpha=1,  
            edgecolor='k', 
            s=50
        )
        plt.colorbar(scatter, label='Zugfestigkeit in [MPa]')
    else:  # Klassen
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = label_train == i
            plt.scatter(X_train_projected[mask, 0], 
                        X_train_projected[mask, 1], 
                        c=color, 
                        label=label, 
                        alpha=0.7, 
                        edgecolor='k', 
                        s=50)
        plt.legend(fontsize=10, title_fontsize=12, loc='best', framealpha=0.8)
    
    plt.title('t-SNE: 2-D-Projektion der Zeitreihe\nfür die '+feature+' und '+graph, fontsize=14, weight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    #plt.tight_layout()
    plt.show()


def plotIQRcurve(kurven_df, feature):
    """ plot feature with IQR for different labels """ 

    df = kurven_df[['key', 'label_name', 'ms', feature]]
    df_grp_med = df.groupby(['label_name', 'ms']).agg({ feature: ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]}).reset_index()
    df_grp_med.columns = ['label_name', 'ms', 'p_med', 'quantil_25', 'quantil_75']

    plt.figure(figsize=(12, 6))

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
    plt.title(f"{feature_txt.split()[0]} im Zeitverlauf mit IQR-Bereich für verschiedene Klassen", fontsize=14, weight='bold', y=1.02)
    plt.xlabel("Zeit in [ms]", fontsize=12)
    plt.xticks(np.arange(0,2750,250))
    plt.ylabel(feature_txt, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_params(estimator, param_x, param_y):
    """ plot params for estimator
       
    - Die Farben zeigen, welche Kombinationen das beste R² liefern         
    - Falls bestimmter Bereich gute Werte hat -> Grid Search
    """
    results = pd.DataFrame(estimator.cv_results_)
    results = results.sort_values(by="mean_test_score", ascending=False)

    sns.scatterplot(data=results, x=param_x, y=param_y, hue="mean_test_score", palette="viridis", size="mean_test_score")
    plt.xscale("log") 
    plt.yscale("log") 
    plt.title("Einfluss von "+ param_x +" & "+ param_y + " auf R²", weight='bold')
    #plt.xlabel("C-Wert")
    #plt.ylabel("Gamma-Wert")
    plt.show()