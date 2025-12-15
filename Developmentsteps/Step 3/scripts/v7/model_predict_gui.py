import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

sys.path.insert(1, '../')

import evaluation
import data_preperation

# Modell und Skalierer laden
def load_models_and_scalers():
    models = {}
    
    try:
        # Klassifikationsmodelle
        models["classifier"] = {
            "model": joblib.load("models/model_SVC_cls.pkl"),
            "pca": joblib.load("models/pca_model_cls.pkl"),
            "scaler_X": joblib.load("models/scaler_x_cls.pkl"),
            "label_encoder": joblib.load("models/label_encoder_cls.pkl")
        }
    except Exception as e:
        models["classifier"] = None
        print(f"Warnung: Klassifikationsmodell konnte nicht geladen werden ({e})")
    
    try:
        # Regressionsmodelle
        models["regressor"] = {
            "model": joblib.load("models/model_SVR_reg.pkl"),
            "pca": joblib.load("models/pca_model_reg.pkl"),
            "scaler_X": joblib.load("models/scaler_x_reg.pkl"),
            "scaler_y": joblib.load("models/scaler_y_reg.pkl")
        }
    except Exception as e:
        models["regressor"] = None
        print(f"Warnung: Regressionsmodell konnte nicht geladen werden ({e})")
    
    return models

# Vorhersagen für alle Dateien in einem Ordner treffen
def predict_from_folder(folder_path):
    models = load_models_and_scalers()
    predictions = {}
    aggregated_features = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = pd.read_csv(file_path)
                X, _ = data_preperation.prepare_data(data)
                
                pred_class, pred_reg = "-", "-"
                
                # ----------------------------------------
                # Klassifikationsmodell (falls vorhanden)
                # ----------------------------------------
                if models["classifier"]:
                    cls_model = models["classifier"]
                    
                    # PCA-Transformation (gleiche Dimension wie beim Training)
                    n_pcs_cls = cls_model["pca"].n_components_
                    X_pca_cls_df = pd.DataFrame(
                        cls_model["pca"].transform(X),
                        columns=[f"PCA{i+1}" for i in range(n_pcs_cls)]
                    )

                    # Skalierung mit denselben Spaltennamen
                    X_scaled_cls = cls_model["scaler_X"].transform(X_pca_cls_df)

                    # Erstellung dataframe
                    X_scaled_cls_df = pd.DataFrame(
                        X_scaled_cls, 
                        columns=X_pca_cls_df.columns
                    )
                    
                    # Klassifikation
                    pred_cls = cls_model["model"].predict(X_scaled_cls_df)
                    pred_class = cls_model["label_encoder"].inverse_transform(pred_cls)[0]
                
                # ----------------------------------------
                # Regressionsmodell (falls vorhanden)
                # ----------------------------------------
                if models["regressor"]:
                    reg_model = models["regressor"]
                    
                    # PCA-Transformation
                    n_pcs_reg = reg_model["pca"].n_components_
                    X_pca_reg_df = pd.DataFrame(
                        reg_model["pca"].transform(X),
                        columns=[f"PCA{i+1}" for i in range(n_pcs_reg)]
                    )
                    
                    # Skalierung
                    X_scaled_reg = reg_model["scaler_X"].transform(X_pca_reg_df)

                    # Erstellung dataframe
                    X_scaled_reg_df = pd.DataFrame(
                        X_scaled_reg, 
                        columns=X_pca_reg_df.columns
                    )
                    
                    # Regression
                    pred_scaled = reg_model["model"].predict(X_scaled_reg_df)
                    pred_reg = reg_model["scaler_y"].inverse_transform(
                        pred_scaled.reshape(-1, 1)
                    ).flatten()[0]
                
                # Vorhersagen speichern
                predictions[file_name] = (pred_class, pred_reg)
                
                # Erstes Zeilen-Dict speichern
                if not X.empty:
                    aggregated_features[file_name] = X.iloc[0].to_dict()
                    
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Verarbeitung von {file_name}: {str(e)}")
    
    return predictions, aggregated_features


# Datei- und Vorhersageanzeige aktualisieren
def update_prediction_display(file_name):
    if file_name in predictions_dict:
        pred_class, pred_reg = predictions_dict[file_name]
        class_label.config(text=f"Vorhergesagte Klasse: {pred_class}")
        prediction_label.config(text=f"Zugkraft: {pred_reg}")
        feature_text.set("\n".join([f"{k}: {v:.4f}" for k, v in features_dict[file_name].items()]))
        #plot_z_values()

# Diagramm für Z-Werte als Scatterplot zeichnen
def plot_z_values():
    if not predictions_dict:
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    files = list(predictions_dict.keys())
    values = [val[1] for val in predictions_dict.values()]
    ids = list(range(1, len(files) + 1))  # IDs als x-Achsen-Werte
    
    ax.scatter(range(len(files)), values, color="#007ACC", s=100)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("Zugkraft")
    ax.set_xlabel("Datei ID")
    ax.set_title("Zugkraft-Vorhersagen für alle Dateien")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    for widget in graph_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
# CSV-Ergebnisse speichern
def save_results():
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Dateien", "*.csv")])
    if save_path:
        results_df = pd.DataFrame([(file, pred) for file, pred in predictions_dict.items()], columns=["Datei", "Predicted_Z"])
        results_df.to_csv(save_path, index=False)
        messagebox.showinfo("Gespeichert", f"Ergebnisse wurden gespeichert: {save_path}")

def open_folder():
    folder_path = filedialog.askdirectory(title="Wähle einen Ordner mit CSV-Dateien")
    if folder_path:
        global predictions_dict, features_dict
        predictions, aggregated_features = predict_from_folder(folder_path)
        predictions_dict = dict(predictions)
        features_dict = aggregated_features
        for widget in file_listbox.winfo_children():
            widget.destroy()
        
        # Scrollbare Liste mit Buttons erstellen
        canvas = tk.Canvas(file_listbox)
        scrollbar = tk.Scrollbar(file_listbox, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for idx, file_name in enumerate(predictions_dict.keys(), start=1):
            frame = tk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(frame, text=f"{idx}", width=5, anchor="e")  # Rechtsbündige ID
            label.pack(side="left")
            
            btn = tk.Button(frame, text=file_name, command=lambda f=file_name: update_prediction_display(f), relief=tk.RAISED, padx=5, pady=2, bg="#f8f8f8")
            btn.pack(side="right", fill=tk.X, expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        plot_z_values()

# Hauptfenster
root = tk.Tk()
root.title("USS Vorhersage")
root.geometry("1024x768")

# Header mit Uni-Logo (optional)
header_frame = tk.Frame(root, bg="#123375", height=80)
header_frame.pack(fill=tk.X)
header_label = tk.Label(header_frame, text="USW Prediction model", font=("Arial", 18, "bold"), fg="white", bg="#003366")
header_label.pack(pady=20)

# Rahmen für den Hauptinhalt
main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill=tk.BOTH, expand=True)

# Linkes Frame für Buttons und Dateiliste
left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

# Frame für Ordner öffnen & Speichern (gemeinsam oben links)
button_frame = tk.Frame(left_frame)
button_frame.pack(fill=tk.X)

# Beide Buttons nebeneinander, teilen sich den Platz gleichmäßig
btn_open = tk.Button(
    button_frame,
    text="📂 open",
    command=open_folder,
    bg="#123375",
    fg="white",
    font=("Arial", 12)
)
btn_open.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)

btn_save = tk.Button(
    button_frame,
    text="💾 save",
    command=save_results,
    bg="#005F50",
    fg="white",
    font=("Arial", 12)
)
btn_save.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)

# Dateiliste direkt unter den Buttons
file_listbox = tk.Frame(left_frame)
file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Rechtes Frame für Vorhersage und Grafik
right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)

# Vorhersage-Labels (oben rechts)
prediction_label = tk.Label(right_frame, text="Zugkraft: -", font=("Arial", 14), anchor="w", justify="left")
prediction_label.pack(anchor="nw", fill=tk.X)

class_label = tk.Label(right_frame, text="Klasse: -", font=("Arial", 14), anchor="w", justify="left")
class_label.pack(anchor="nw", pady=10, fill=tk.X)

# Label für weitere Features
feature_text = tk.StringVar()
feature_label = tk.Label(right_frame, textvariable=feature_text, font=("Arial", 12), anchor="w", justify="left")
feature_label.pack(anchor="nw", pady=10, fill=tk.X)

# Grafik unter den Vorhersage-Labels
graph_frame = tk.Frame(right_frame)
graph_frame.pack(anchor="nw", fill=tk.BOTH, expand=True, pady=20)

# Dictionaries (falls benötigt für spätere Logik)
predictions_dict = {}
features_dict = {}

root.mainloop()