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