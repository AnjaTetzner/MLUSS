import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import evaluation
import data_preperation

# Modell und Skalierer laden
def load_model_and_scalers():
    # Das gespeicherte Modell
    model = joblib.load("models/model_SVR_reg.pkl")
    # Die gespeicherte PCA-Transformation
    pca = joblib.load("models/pca_model_reg.pkl")
    # Skalierung für X
    scaler_X = joblib.load("models/scaler_x_reg.pkl")  
     # Skalierung für y
    scaler_y = joblib.load("models/scaler_y_reg.pkl") 
    return model, pca, scaler_X, scaler_y

# Vorhersagen für alle Dateien in einem Ordner treffen
def predict_from_folder(folder_path):
    model, pca, scaler_X, scaler_y = load_model_and_scalers()
    predictions = []
    aggregated_features = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = pd.read_csv(file_path)
                X, _ = data_preperation.prepare_data(data)
                X_pca = pd.DataFrame(pca.transform(X), columns=[f"PCA{i+1}" for i in range(pca.n_components_)])
                X_scaled = pd.DataFrame(scaler_X.transform(X_pca), columns=X_pca.columns)
                pred_scaled = model.predict(X_scaled)
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
                predictions.append((file_name, pred))
                aggregated_features[file_name] = X.iloc[0].to_dict()
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Verarbeitung von {file_name}:{str(e)}")
    return predictions, aggregated_features

# Datei- und Vorhersageanzeige aktualisieren
def update_prediction_display(file_name):
    if file_name in predictions_dict:
        prediction_label.config(text=f"Vorhersage: {predictions_dict[file_name]:.2f}")
        feature_text.set("\n".join([f"{k}: {v:.4f}" for k, v in features_dict[file_name].items()]))
        plot_z_values()

# Diagramm für Z-Werte als Scatterplot zeichnen
def plot_z_values():
    if not predictions_dict:
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    files = list(predictions_dict.keys())
    values = list(predictions_dict.values())
    
    ax.scatter(range(len(files)), values, color="#007ACC", s=100)
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=45, ha="right")
    ax.set_ylabel("Predicted Z-Value")
    ax.set_title("Vorhersagen für alle Dateien")
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

# Ordner auswählen
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
        
        for file_name in predictions_dict.keys():
            btn = tk.Button(scrollable_frame, text=file_name, command=lambda f=file_name: update_prediction_display(f), relief=tk.RAISED, padx=5, pady=2, bg="#f8f8f8")
            btn.pack(fill=tk.X, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        plot_z_values()

# GUI erstellen
root = tk.Tk()
root.title("ML Vorhersage Tool")
root.geometry("900x600")

# Header mit Uni-Logo
header_frame = tk.Frame(root, bg="#003366", height=80)
header_frame.pack(fill=tk.X)
header_label = tk.Label(header_frame, text="ML Vorhersage Tool", font=("Arial", 18, "bold"), fg="white", bg="#003366")
header_label.pack(pady=20)

# Hauptlayout
main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill=tk.BOTH, expand=True)

# Frame für Vorhersageanzeige
prediction_frame = tk.Frame(main_frame)
prediction_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Label für die Vorhersage
prediction_label = tk.Label(prediction_frame, text="Zugkraft: -", font=("Arial", 14))
prediction_label.pack(pady=10)

# Features Frame
feature_text = tk.StringVar()
feature_label = tk.Label(prediction_frame, textvariable=feature_text, font=("Arial", 12), justify="left")
feature_label.pack(pady=10)

# Frame für Vorhersage und Diagramm
output_frame = tk.Frame(main_frame)
output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Frame für das Diagramm
graph_frame = tk.Frame(output_frame)
graph_frame.pack(pady=20)


# Datei-Liste auf der linken Seite
file_list_frame = tk.Frame(main_frame, width=200, bg="#f0f0f0")
file_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

# Ordner öffnen Button (zentriert über der Liste)
btn_open = tk.Button(file_list_frame, text="📂 Ordner öffnen", command=open_folder, bg="#007ACC", fg="white", font=("Arial", 12))
btn_open.pack(pady=10, fill=tk.X)

# Datei-Buttons anzeigen
file_listbox = tk.Frame(file_list_frame)
file_listbox.pack(fill=tk.BOTH, expand=True)

# Ergebnisse speichern (unten rechts)
btn_save = tk.Button(root, text="💾 Ergebnisse speichern", command=save_results, bg="#28A745", fg="white", font=("Arial", 12))
btn_save.pack(side=tk.BOTTOM, pady=10, anchor="se")

predictions_dict = {}
features_dict = {}

root.mainloop()
