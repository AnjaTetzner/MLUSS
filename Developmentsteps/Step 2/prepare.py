'''
Hilfsfunktionen zum Einlesen der Schweißdaten
'''

import io, os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tsgm

def read_special_csv(path):
    '''
    Einlesen der Maschinendaten.
    Vorspann muss entfernt werden, CSV beginnt nach Zeile #CSV-START#
    '''
    res = list()
    flag = False
    for line in open(path):
        if line.strip() == '#CSV-START#':
            flag = True
            continue
        if flag:
            res.append(line)
    #print('\n'.join(res)[:20])
    #return '\n'.join(res)
    stream = io.StringIO('\n'.join(res))
    df = pd.read_csv(stream, sep=';')
    df.columns = ['ms', 'power', 'force', 'dist']
    return df
    

def read_kurven(data_dir, kurven_dict, prefix):
    '''
    Einlesen aller Schweisskurven einer Gruppe.
    Verzeichnis ist {data_dir}
    Kurve wird als DataFrame im Dictionary kurven_dict
    mit Key {prefix}_{nr} gespeichert.
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



def check_completeness(keys, kurven):
    '''
    Prüft, ob alle Schweißkurven der in der Zugversuchstabelle aufgeführten
    Werte vorhanden sind.
    Nur Fehlerausgabe, keine Behandlung
    '''
    for key in keys:
        if key not in kurven:
            print("Kurve {key} fehlt")

def read_all(base_dir, data_dirs):
    '''
    Einlesen aller Daten.
    base_dir: Hauptverzeichnis Daten
    data_dirs: Unterverzeichnisse (= Gruppe) der Versuche (ok, fehler_xxx)
               Jedes enthält CSV-Datei zugversuch.csv und Verzeichnis schweisskurven
    return: DataFrame aller Zugversuche und Dictionary (key=versuchs_id, value=DataFrame Schweißkurve)
    '''
    kurven = dict()
    zugversuche = list()
    # target labels for fail data
    for idx, data_dir in enumerate(data_dirs):
        zugversuch = pd.read_csv(os.path.join(base_dir, data_dir, 'zugversuch.csv'), sep=';', decimal=',')
        zugversuch.columns = ['nr', 'festigkeit']
        prefix = data_dir + '_'
        zugversuch['nr'] = zugversuch.nr.astype('str')
        zugversuch['key'] = prefix + zugversuch.nr
        zugversuch['label'] = idx
        zugversuch['label_text'] = data_dir
        read_kurven(os.path.join(base_dir, data_dir, 'schweisskurven'), kurven, prefix)
        zugversuche.append(zugversuch)
        check_completeness(zugversuch.key, kurven)
    zugversuche = pd.concat(zugversuche, ignore_index=True)
    return zugversuche, kurven

# Subsampling 1: Kurve auf max. Länge beschneiden, dann Subsampling
class Subsample1(BaseEstimator, TransformerMixin):
    def __init__(self, ts_len, subsample):
        self.ts_len = ts_len
        self.subsample = subsample

    def fit(self):
        return self
    
    def pd_to_array(self, df):
        '''
        Umwandlung DataFrame der Schweißkurve in Array
        Die Länge wird auf {ts_len} gekürzt.
        Jeder {subsample}-te Punkt wird ausgewählt.
        Aus der Feature-Dimension wird die Zeit entfernt.
        Es findet kein Padding statt!
        '''
        a = df.values
        #print(a.shape)
        return a[:self.ts_len:self.subsample,1:].copy()

    def transform(self, keys, kurven):
        '''
        Verketten aller Schweißkurven-Arrays zu 3D-Tensor
        (versuch, zeitpunkte, features).
        versuche: DataFrame der Versuche mit Spalte {key}
        kurven: Dict {key}: DataFrame Schweißkurve
        ts_len: Länge der Zeitreihe
        subsample: Subsample Rate (nach Beschneiden auf ts_len)
        '''
        st = list()
        for key in keys:
            # add axis for versuch_nr
            a = np.expand_dims(self.pd_to_array(kurven[key]), axis=0)
            st.append(a)
        return np.vstack(st)

# Alternatives Subsampling: Kurvenlänge nun beliebig, es werden k gleichlange Zeitschritte entsprechend
# Kurvenlänge gewählt
class Subsample2(BaseEstimator, TransformerMixin):
    def __init__(self, subsample):
        self.subsample = subsample

    def fit(self):
        return self
    
    def subsample_df(self, kurve):
        '''
        kurve: DataFrame
        result: array(subsample, 3)
        '''
        a = kurve.values[:, 1:]
        n = a.shape[0]
        step = n//self.subsample
        #print(n, step, step*subsample)
        res = a[:step*self.subsample:step,:].copy()
        #print(res.shape)
        return res

    def transform(self, keys, kurven):
        '''
        Umwandeln aller Schweißkurven der Versuche in
        Array [versuche, zeitpunkte, features]
        '''
        st = list()
        for key in keys:
            # add axis for versuch_nr
            a = np.expand_dims(self.subsample_df(kurven[key]), axis=0)
            #print(a.shape)
            st.append(a)
        return np.vstack(st)

class StatFeatures(BaseEstimator, TransformerMixin):
    '''
    Umwandeln der Zeitreihe in statistische Features
    '''
    def __init__(self):
        pass
    def fit(self):
        return self
    def transform(self, keys, kurven):
        '''
        Umwandeln der Schweißkurven der Versuche in
        Array[versuche, features]
        '''
        st = list()
        for key in keys:
            kurve = kurven[key]
            p_mean = kurve.power.mean()
            p_median = kurve.power.median()
            p_max = kurve.power.max()
            p_std = kurve.power.std()
            f_mean = kurve.force.mean()
            f_median = kurve.force.median()
            f_max = kurve.force.max()
            f_std = kurve.force.std()
            d_mean = kurve.dist.mean()
            d_median = kurve.dist.median()
            d_max = kurve.dist.max()
            d_std = kurve.dist.std()
            ti = kurve.power.count()
            st.append(np.array([p_mean, p_median, p_max, p_std,
                                f_mean, f_median, f_max, f_std,
                                d_mean, d_median, d_max, d_std,
                                ti]))
        return np.stack(st)
            
class StatFeatures2(BaseEstimator, TransformerMixin):
    '''
    Umwandeln der Zeitreihe in statistische Features, jetzt mit Quartilen
    '''
    def __init__(self, q=[0.25, 0.5, 0.75, 1.0]):
        '''
        Definieren der Quantile.
        Default: Quartile ohne min.
        '''
        self.q = q
        
    def fit(self):
        return self
    def transform(self, keys, kurven):
        '''
        Umwandeln der Schweißkurven der Versuche in
        Array[versuche, features]
        '''
        st = list()
        for key in keys:
            kurve = kurven[key]
            p_mean = kurve.power.mean()
            p_q = np.quantile(kurve.power, self.q)
            p_std = kurve.power.std()
            f_mean = kurve.force.mean()
            f_q = np.quantile(kurve.force, self.q)
            f_std = kurve.force.std()
            d_mean = kurve.dist.mean()
            d_q = np.quantile(kurve.dist, self.q)
            d_std = kurve.dist.std()
            ti = kurve.power.count()
            features = [p_mean] + list(p_q) + [p_std] + \
                       [f_mean] + list(f_q) + [f_std] + \
                       [d_mean] + list(d_q) + [d_std] + \
                       [ti] # p_q etc. sind np-Arrays, wir brauchen aber list
            st.append(np.array(features))
        return np.stack(st)

class StatFeatures3(BaseEstimator, TransformerMixin):
    '''
    Umwandeln der Zeitreihe in statistische Features, jetzt mit Quartilen
    Nun mit Knicken:
    - ti_b50: Zeitpunkt, an dem erstmals 50% des Maximums erreicht werden
    - ti_b75: Zeitpunkt, an dem erstmals 75% des Maximums erreicht werden
    '''
    def __init__(self, q=[0.25, 0.5, 0.75, 1.0]):
        '''
        Definieren der Quantile.
        Default: Quartile ohne min.
        '''
        self.q = q
        
    def fit(self):
        return self
    def transform(self, keys, kurven):
        '''
        Umwandeln der Schweißkurven der Versuche in
        Array[versuche, features]
        '''
        st = list()
        for key in keys:
            kurve = kurven[key]
            p_mean = kurve.power.mean()
            p_q = np.quantile(kurve.power, self.q)
            p_std = kurve.power.std()
            p_max = np.max(kurve.power)
            p_b50 = kurve.loc[kurve.power >= 0.50*p_max].iloc[0].ms
            p_b75 = kurve.loc[kurve.power >= 0.75*p_max].iloc[0].ms
            f_mean = kurve.force.mean()
            f_q = np.quantile(kurve.force, self.q)
            f_std = kurve.force.std()
            f_max = np.max(kurve.force)
            f_b50 = kurve.loc[kurve.force >= 0.50*f_max].iloc[0].ms
            f_b75 = kurve.loc[kurve.force >= 0.75*f_max].iloc[0].ms
            d_mean = kurve.dist.mean()
            d_q = np.quantile(kurve.dist, self.q)
            d_std = kurve.dist.std()
            d_max = np.max(kurve.dist)
            d_b50 = kurve.loc[kurve.dist >= 0.50*d_max].iloc[0].ms
            d_b75 = kurve.loc[kurve.dist >= 0.75*d_max].iloc[0].ms
            ti = kurve.ms.max()
            features = [p_mean] + list(p_q) + [p_std, p_b50, p_b75] + \
                       [f_mean] + list(f_q) + [f_std, f_b50, f_b75] + \
                       [d_mean] + list(d_q) + [d_std, d_b50, d_b75] + \
                       [ti] # p_q etc. sind np-Arrays, wir brauchen aber list
            st.append(np.array(features))
        return np.stack(st)
            
class TsMaxScaler(BaseEstimator, TransformerMixin):
    '''
    MinMax-Scaler für jedes Feature über alle Zeitreihen
    '''
    def __init__(self):
        self.max = None
    def fit(self, X, y=None):
        self.max = X.max(axis=(0,1), keepdims=True)
        return self
    def transform(self, X):
        return (X/self.max).copy()

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