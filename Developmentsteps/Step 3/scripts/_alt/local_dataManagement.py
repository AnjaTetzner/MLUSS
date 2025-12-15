import pandas as pd

# Definiere eine Klasse, um die Daten zu verwalten
class DataStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Lädt die Daten aus der CSV-Datei."""
        self.data = pd.read_csv(self.file_path)

    def get_data(self):
        """Gibt die geladenen Daten zurück."""
        if self.data is None:
            raise ValueError("Daten wurden noch nicht geladen. Verwende 'load_data', um sie zu laden.")
        return self.data