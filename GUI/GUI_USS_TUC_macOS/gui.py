import os
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk 
from tkinter import ttk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from tkmacosx import Button
import gui_helper

os.environ['TK_SILENCE_DEPRECATION'] = '1' 

label_names = ['OK-Schweißung', 'NEAR-OK-Sonotrodenwechsel', 'Öl auf Terminalversatz', 'Leitungsversatz', 'Terminalversatz']
DATA = ['ok', 'near_ok_sonowechsel', 'fehler_oel', 'fehler_leitungsversatz', 'fehler_terminversatz']
label  = dict() 
for idx, fname in enumerate(DATA):
    label[fname] = idx

MODEL_DIR = "GUI"

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML USS")
        self.root.geometry("1500x1000")
        self.root.configure(bg="white")

        self.active_button = None

        # Aufteilung der GUI in drei Bereiche
        ###################################################
        #                 ÜBERSCHRIFT                     #
        ###################################################
        self.frame_title = tk.Frame(root, width=1500, height=120, bg="#003366")
        self.frame_title.pack(side="top", padx=0, pady=0, fill="x")# expand=True, fill="both",

        # Wichtig: Pack-Propagation deaktivieren, damit die Höhe nicht überschrieben wird!
        self.frame_title.pack_propagate(False)
        
        # Logo TU Chemnitz
        img_path = os.path.join("tuc_logo.png")
        if not os.path.exists(img_path):
            print(f"Bild '{img_path}' nicht gefunden!")
            return
        image = Image.open(img_path)
        image = image.resize((201, 105), Image.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(image)
        # Frame für Bild
        content_frame = tk.Frame(self.frame_title, bg="#003366")
        content_frame.pack(fill="both", expand=True)
        img_label = tk.Label(content_frame, image=self.image_tk, bg="#003366", cursor="hand")
        img_label.pack(side="left", padx=37, pady=5)
        img_label.bind("<Button-1>", lambda event: self.show_instructions())
        def on_enter_img(event): img_label.config(bg="#004080")  # etwas helleres Blau
        def on_leave_img(event): img_label.config(bg="#003366")  # Originalfarbe zurücksetzen
        img_label.bind("<Enter>", on_enter_img)
        img_label.bind("<Leave>", on_leave_img)

        # Vertikaler weißer Strich
        separatorV = tk.Frame(content_frame, bg="white", width=2, height=80)
        separatorV.pack(side="left", padx=5)

        # Textfelder
        title_label1 = tk.Label(content_frame, text="Professur Verbundwerkstoffe und Werkstoffverbunde & Wirtschaftsinformatik 2",
                               font=("Arial", 14, "bold"), fg="white", bg="#003366")
        title_label1.pack(side="top", anchor="w", padx=20, pady=(25, 10))
        title_label2 = tk.Label(content_frame, text="Machine Learning Projekt zur Vorhersage von Zugfestigkeiten beim Ultraschallschweißen",
                               font=("Arial", 18, "bold"), fg="white", bg="#003366")
        title_label2.pack(side="top", anchor="w", padx=20)

        ###################################################
        #                 MENÜ LINKS                      #
        ###################################################
        self.frame_menu = tk.Frame(root, width=100, height=1000, bg="#003366") 
        self.frame_menu.pack(side="left", padx=0, pady=0, fill="y")

        # Horizontaler weißer Strich
        #separatorH = tk.Frame(self.frame_menu, bg="white", width=300, height=2)
        #separatorH.pack(side="top", pady=0)#, fill="x")
        # Gemeinsame Button-Eigenschaften
        button_kwargs = {
                "font": ("Arial", 18, "bold"),
                "fg": "white",
                "bg": "#003366", 
                "relief": "flat", # flat, groove, raised, ridge, solid, or sunken
                "borderwidth": 0,
                "borderless":1,
                #"highlightthickness": 0,
                #"highlightbackground":"#003366",
                "activebackground": "white",
                "activeforeground": "black",
                "cursor": "hand",
                "height": 50,
                "width": 20,
                "justify": "center",
                #"state":"active",
        }
        padx_buttons = 15
        pady_buttons = 20

        # Dropdown-Menü "Maschine auswählen"          
        self.machine_var = tk.StringVar()
        self.machine_var.set("⚙️ Maschine auswählen")
        self.machines = self.get_machines()
        self.machine_menu = tk.OptionMenu(self.frame_menu, self.machine_var, *self.machines, command=lambda: [self.highlight_button(self.machine_menu), self.update_font()])
        self.machine_menu.config(height=1, width=19, bg="#003366", fg="white", font=("Arial", 18, "bold"), cursor="hand") 
        self.machine_menu.pack(padx=padx_buttons, pady=(5,25), fill="both") 

        # Button "Datei-Auswahl" => Laden der Testdaten  
        self.button_data = Button(self.frame_menu, text="📂 Daten auswählen", command=lambda: [self.highlight_button(self.button_data), self.load_csv()], **button_kwargs)
        self.button_data.pack(padx=padx_buttons, pady=pady_buttons, fill="both")  
        self.add_hover_effect(self.button_data)

        # Button "Datenanalyse"
        self.button_analyse = Button(self.frame_menu, text="📊 Datenanalyse", command=lambda: [self.highlight_button(self.button_analyse), self.run_data_analysis()], **button_kwargs)
        self.button_analyse.pack(padx=padx_buttons, pady=pady_buttons, fill="both")
        self.add_hover_effect(self.button_analyse)

        # Button "Vorhersage"
        self.button_pred = Button(self.frame_menu, text="🔮 Modellvorhersage", command=lambda: [self.highlight_button(self.button_pred), self.run_prediction()], **button_kwargs)
        self.button_pred.pack(padx=padx_buttons, pady=pady_buttons, fill="both")
        self.add_hover_effect(self.button_pred)

        # Button "Modelltest"
        self.button_test = Button(self.frame_menu, text="🎯 Modelltest", command=lambda: [self.highlight_button(self.button_test), self.run_model_test()], **button_kwargs)
        self.button_test.pack(padx=padx_buttons, pady=pady_buttons, fill="both")
        self.add_hover_effect(self.button_test)

        # TODO Button "Auswahl einzelner Kurven"
        self.button_curve = Button(self.frame_menu, text="📉 Kurvenauswahl", command=lambda: [self.highlight_button(self.button_curve), self.select_curve()], **button_kwargs)
        self.button_curve.pack(padx=padx_buttons, pady=pady_buttons, fill="both")
        self.add_hover_effect(self.button_curve)

        # TODO Button "Ergebnisse speichern" 
        self.button_save = Button(self.frame_menu, text="💾 Ergebnisse speichern", command=lambda: [self.highlight_button(self.button_save), self.save_files()], **button_kwargs)
        self.button_save.pack(padx=padx_buttons, pady=pady_buttons, fill="both")
        self.add_hover_effect(self.button_save)

        # Datum unten im Menüband anzeigen
        self.date_label = tk.Label(self.frame_menu, text=datetime.datetime.now().strftime("%d.%m.%Y"),
                                   font=("Arial", 10), bg="#003366", fg="white")
        self.date_label.pack(side="bottom", pady=10)

        ###################################################
        #             RECHTES ANZEIGEFENSTER              #
        ###################################################
        self.frame_show = tk.Frame(root, width=300, height=800, bg="white")
        self.frame_show.pack(side="top", fill="both", expand=True)#, padx=5, pady=5
        self.show_instructions()
     
        self.svc_model = None
        self.svr_model = None
        self.csv_file = None
        

    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    def show_instructions(self):
        # Aktiven Button zurücksetzen
        if self.active_button:
            self.active_button.config(bg="#003366")
            self.active_button = None

        # Vorherige Inhalte im frame_show löschen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

        # Anleitungstitel
        self.title1 = tk.Label(self.frame_show, text="Anleitung des Programms", font=("Arial", 16, "bold"), bg="white", fg="#61686A", anchor="w")
        self.title1.pack(side="top", fill="x", padx=18, pady=5, ipadx=10, ipady=10)

        # Anleitungstext
        self.title2 = tk.Label(
            self.frame_show,
            text=(
                " 1️⃣  Maschine auswählen, auf der die Schweißdaten aufgezeichnet wurden\n\n"
                
                " 2️⃣  CSV-Datei mit den Daten auswählen, die für die Vorhersage genutzt werden sollen\n\n\n\n"
                
                " 📊  Datenanalyse:\n"
                "     ➢ Überblick über die eingelesenen Daten\n\n"
                
                " 🔮  Vorhersage:\n"
                "     ➢ Vorhersage der Zugfestigkeiten\n"
                "     ➢ Optionale Vorhersage der Klassen (falls in den Trainingsdaten vorhanden)\n\n"
                
                " 🎯  Modelltest:\n"
                "     ➢ Vergleich der Vorhersageergebnisse mit der Realität, \n"
                "       falls Zugfestigkeit und/oder Klasse in den Testdaten gegeben sind\n\n"
                
                " 💾  Ergebnisse speichern:\n"
                "     ➢ Tabellen als CSV und Diagramme als PNG speichern\n"
                "     ➢ Zielordner: GUI_USS_TUChemnitz/GUI/[Maschine]/Ergebnisse\n\n"
                
                " 🖼️  Zurück zur Anleitung:\n"
                "     ➢ ein Klick auf das TU-Logo bringt dich zurück zu dieser Übersicht\n\n"
                
                " 🆕  Modellanwendung auf einer neuen Maschine:\n"
                "     ➢ Neuen Ordner anlegen unter: GUI_USS_TUChemnitz/GUI/[NeueMaschine]\n"
                "     ➢ Unterordner 'Trainingsdaten' erstellen\n"
                "         (jede Schweißkurve als einzelne CSV-Datei, benannt nach ihrer Nummer)\n"
                "     ➢ Schweißkurven für die Vorhersage in den Ordner 'Testdaten' legen\n"
                "     ➢ CSV-Datei mit den Spalten 'nr' und optional 'klasse' und 'zugfestigkeit' erstellen\n"
                "     ➢ Falls reale Messwerte vorhanden sind, kann über 'Modelltest' \n"
                "         das Vorhersagemodell mit den gemessenen Werten verglichen werden\n\n"
            ),
            font=("Arial", 14), bg="white", fg="#61686A", anchor="w", justify="left"
        )
        self.title2.pack(side="top", fill="x", padx=55, pady=5, ipadx=10, ipady=10)

    def highlight_button(self, clicked_button):
        # Vorherigen aktiven Button zurücksetzen 
        if self.active_button and self.active_button != clicked_button:
            self.active_button.config(bg="#003366")  
        # Geklickten Button hervorheben
        clicked_button.config(bg="#004080")  
        # Aktiven Button merken
        self.active_button = clicked_button

    def add_hover_effect(self, button, enter_color="#004080", leave_color="#003366"):
        """ Beim Hovern wird die Farbe von dunkelblau zu hellblau gesetzt. """
        def on_enter(event): 
            button.config(bg=enter_color)
        def on_leave(event): 
            button.config(bg=leave_color)
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def get_machines(self):
        """ Listet die verfügbaren Maschinen auf """
        if not os.path.exists(MODEL_DIR):
            messagebox.showerror("Fehler", f"Modelldatei-Verzeichnis '{MODEL_DIR}' nicht gefunden")
            return []
        return [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    
    def update_font(self, selected_value):
        """ Ändert die Schriftart auf fett nach Auswahl """ 
        self.machine_menu.config(font=("Arial", 18, "bold")) 

    def load_csv(self):
        """ Öffnet den Datei-Dialog und aktualisiert den Button-Text mit dem Dateinamen in Fett """
        # Alten Inhalt aus self.frame_show entfernen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

        self.csv_file = filedialog.askopenfilename(filetypes=[("CSV-Dateien", "*.csv")])
        if self.csv_file: 
            filename = self.csv_file.split("/")[-1] 
            self.button_data.config(text=filename, font=("Arial", 13, "bold"), height=50, width=250, fg="white", bg="#003366", borderless=1, justify="center")
    
    def load_models(self, modelname):
        """ Lädt die ML-Modelle basierend auf der Maschinenwahl """
        machine = self.machine_var.get()
        if not machine:
            messagebox.showerror("Fehler", "Bitte eine Maschine auswählen!")
            return False
        
        model_path = os.path.join(MODEL_DIR, machine, modelname+"_stat.joblib")

        if not os.path.exists(model_path):
            if modelname=="svr":
                messagebox.showerror("Fehler", f"'{modelname}'-Modell für '{machine}' nicht gefunden.")
            return False
        
        self.model = joblib.load(model_path)

        return True
    
    def read_data(self):
        """ """
        if not self.csv_file:
            messagebox.showerror("Fehler", "Bitte eine CSV-Datei auswählen!")
            return
        
        if self.csv_file:
            # CSV-Datei mit Versuchsnr. laden
            df = pd.read_csv(self.csv_file, sep=';', decimal=',', thousands='.')
            df.columns = df.columns.str.lower() 
            y_true = df['zugfestigkeit'].tolist() if 'zugfestigkeit' in df.columns else None
            klasse_true =   df['klasse'].tolist() if 'klasse'        in df.columns else None
            if 'nr' not in df.columns:
                messagebox.showerror("Fehler", "CSV-Datei muss die Spalte 'nr' enthalten.")
                return
            versuche_test = df['nr'].astype(str).tolist() 
            #if klasse_true:
                #df['key'] = df['klasse'] + '_' + df['nr'] # TODO newclass
            # Daten aus den entsprechenden Versuchsnr.-Dateien laden
            testdata_path = os.path.join("GUI", self.machine_var.get(), "Testdaten")

            kurven = {}
            for nr in versuche_test:
                file_path = os.path.join(testdata_path, f"{nr}.csv")
                if not os.path.exists(file_path):
                    messagebox.showerror("Fehler", f"Datei '{nr}.csv' nicht gefunden!")
                    return
                
                # CSV einlesen mit den Spalten "ms", "Power[W]", "Force[N]", "Distance[mm]"
                df_kurve = read_csv_skip_until_ms(file_path)
                kurven[nr] = df_kurve  # Speichern der Daten für die Versuchsnr.
           
            ## Data Preparation ##
            stat_feat = gui_helper.StatFeatures() # Feature-Extraktion 
            stat_feat.fit()  # Fit wird nicht unbedingt benötigt, aber zur Konsistenz
            X_test = stat_feat.transform(versuche_test, kurven)

        return y_true, klasse_true, df, kurven, X_test


    def read_and_predict(self):
        """ Einlesen der CSV-Datei und Modellvorhersage """
        # Einlesen
        y_true, klasse_true, df, kurven, X_test = self.read_data()

        # Modellvorhersage
        if self.csv_file:
            # Skalierung 
            std_scaler = joblib.load(os.path.join("GUI", self.machine_var.get(), "scaler.joblib"))
            if not std_scaler:
                messagebox.showerror("Fehler", f"'Scaler für '{self.machine_var.get()}' nicht gefunden.")
                return False
            X_test_scaled = std_scaler.transform(X_test)

            # Vorhersage mit SVC-Modell
            if self.load_models("svc"):
                y_pred_svc = self.model.predict(X_test_scaled) 
                y_pred_svc = pd.DataFrame(y_pred_svc)
                y_pred_svc.columns = ['klasse_pred']
                df = pd.concat([df, y_pred_svc], axis=1)

            # Vorhersage mit SVR-Modell
            if not self.load_models("svr"):
                return
            y_pred_svr = self.model.predict(X_test_scaled) 
            y_pred_svr = pd.DataFrame(y_pred_svr)*2000
            y_pred_svr.columns = ['y_pred']
            df = pd.concat([df, y_pred_svr], axis=1)

            # replace small predictions with 0
            df.loc[df['y_pred'] < 10, 'y_pred'] = 0

            return df, y_true, klasse_true
        return
        
    ###################
    #  DATA ANALYSIS  #
    ###################
    def run_data_analysis(self):
        """ Führt die Datenanalyse durch und zeigt zwei Diagramme nebeneinander im Grid """

        # Alten Inhalt aus self.frame_show entfernen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

        self.frame_show.rowconfigure(0, weight=1)
        self.frame_show.rowconfigure(1, weight=9)
        self.frame_show.columnconfigure(0, weight=1)

        # frame_show in oben und unten aufteilen
        frame_oben = tk.Frame(self.frame_show, bg="white")
        frame_oben.grid(row=0, column=0, sticky="nsew")

        frame_unten = tk.Frame(self.frame_show, bg="white")
        frame_unten.grid(row=1, column=0, sticky="nsew")

        # Konfiguration für die Spalten innerhalb von frame_show
        frame_oben.grid_columnconfigure(0, weight=1)
        frame_oben.grid_columnconfigure(1, weight=2)
        frame_unten.grid_columnconfigure(0, weight=1)
        frame_unten.grid_columnconfigure(1, weight=1)
        frame_unten.grid_columnconfigure(2, weight=1)
        frame_unten.grid_rowconfigure(0, weight=2)
        frame_unten.grid_rowconfigure(1, weight=1)

        """
        # frame_show in links und rechts aufteilen
        frame_links = tk.Frame(self.frame_show, bg="white")
        frame_links.grid(row=0, column=0, rowspan=6, sticky="nsew")

        frame_rechts = tk.Frame(self.frame_show, bg="white")
        frame_rechts.grid(row=0, column=1, rowspan=6, sticky="nsew")

        # Konfiguration für die Spalten innerhalb von frame_show
        frame_links.grid_columnconfigure(0, weight=1)
        frame_rechts.grid_columnconfigure(0, weight=4)

        # Konfiguration für die Zeilen innerhalb von frame_show
        frame_links.grid_rowconfigure(2, weight=2)  
        frame_links.grid_rowconfigure(4, weight=2)  
        """
        # Überschrift
        headline = tk.Label(frame_oben, text="Datenanalyse der gegebenen Testdaten", font=("Arial", 18, "bold"),
                            bg="white", fg="black", anchor="w")
        headline.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        # Einlesen
        y_true, klasse_true, df, kurven, X_test = self.read_data()

        if self.csv_file:
            # Informationen anzeigen
            label_versuche = tk.Label(frame_oben, text=f"➜ {df['nr'].nunique()} verschiedene Versuchsnummern",
                                    font=("Arial", 12), bg="white", fg="black", anchor="w")
            label_versuche.grid(row=1, column=0, columnspan=2, sticky="w", padx=25)

            if y_true:
                label_y = tk.Label(frame_unten, text=f"➜ Mittelwert der Zugfestigkeiten: {int(df['zugfestigkeit'].mean())} MPa\n"
                                    f"➜ Minimum von {int(df['zugfestigkeit'].min())} MPa und Maximum von {int(df['zugfestigkeit'].max())} MPa",
                                        font=("Arial", 12), bg="white", fg="black", anchor="w", justify="left")
                label_y.grid(row=1, column=0, columnspan=2, sticky="w", padx=55, pady=(0, 20))
            else:
                label_y = tk.Label(frame_oben, text=f"➜ keine Zugfestigkeiten vorhanden",
                                        font=("Arial", 12), bg="white", fg="black", anchor="w")
                label_y.grid(row=2, column=0, columnspan=2, sticky="w", padx=55, pady=(0, 2))

            if klasse_true:
                label_klassen = tk.Label(frame_unten, text=f"➜ {df['klasse'].nunique()} verschiedene Klassen", 
                                        font=("Arial", 12), bg="white", fg="black", anchor="w")
                label_klassen.grid(row=1, column=1, columnspan=2, sticky="w", padx=55, pady=(0, 20))
            else:
                label_klassen = tk.Label(frame_oben, text=f"➜ keine Klassen vorhanden",
                                        font=("Arial", 12), bg="white", fg="black", anchor="w")
                label_klassen.grid(row=3, column=0, columnspan=2, sticky="w", padx=55, pady=(0, 20))    

            # Zugfestigkeits-Histogramm (links unten)
            if y_true is not None:
                fig1, ax1 = plt.subplots(figsize=(2, 2))
                fig1.tight_layout()
                d = pd.DataFrame(y_true, columns=['zugf'])
                d['zugf'].plot(kind="hist", bins=20, color="#003366", edgecolor="black", alpha=1, ax=ax1)
                ax1.set_title("Histogramm der Zugfestigkeiten", fontsize=10, weight="bold")
                ax1.set_xlabel("Zugfestigkeit [MPa]", fontsize=9)
                ax1.set_ylabel("Häufigkeit", fontsize=9)
                ax1.grid(axis="y", linestyle="--", alpha=0.7)
                canvas1 = FigureCanvasTkAgg(fig1, master=frame_unten)
                canvas1.draw()
                canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=1, pady=(10,2))

            # Klassenverteilung (rechts unten)
            if klasse_true: # is not None:
                fig2, ax2 = plt.subplots(figsize=(2, 2))
                fig2.tight_layout()
                klasse_series = pd.Series(klasse_true)
                counts = klasse_series.value_counts().sort_index()
                counts.plot(kind="bar", color="#003366", edgecolor="black", alpha=1, ax=ax2)
                ax2.set_title("Häufigkeiten der Klassen", fontsize=10, weight="bold")
                ax2.set_xlabel("Klassen", fontsize=9)
                ax2.set_ylabel("Häufigkeit", fontsize=9)
                ax2.grid(axis="y", linestyle="--", alpha=0.7)
                ax2.set_xticks(range(len(counts)))
                ax2.set_xticklabels(counts.index, rotation=0)
                canvas2 = FigureCanvasTkAgg(fig2, master=frame_unten)
                canvas2.draw()
                canvas2.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=1, pady=(10,2))

                # Legende (rechts unten)
                legend_text = "\n".join([f"{i} → {label}" for i, label in enumerate(label_names)])
                legend_label = tk.Label(frame_unten, text=legend_text, font=("Arial", 12), fg="black", bg="white",
                                        justify="left", anchor="w", padx=10, pady=10, relief="solid", borderwidth=0) 
                legend_label.grid(row=0, column=2, sticky="sw", padx=25)

            # Schweißkurvenverläufe (oben)
            kurven_df = gui_helper.get_kurven_df(kurven).reset_index()
            if klasse_true:
                # Für jede Zeile in kurven_df den passenden label_name zusammensetzen # TODO newclass
                df = df.rename(columns={'nr': 'key'})
                kurven_df['key'] = kurven_df['key'].astype(int)
                df['key'] = df['key'].astype(int)
                label_df = pd.DataFrame({'klasse': list(label.values()), 'label_text': list(label.keys())})
                merged = kurven_df.merge(df[['key', 'klasse']], on='key', how='left')
                merged = merged.merge(label_df, on='klasse', how='left')
                merged['key_name'] = merged.apply(lambda row: f"{row['label_text']}_{row['key']}", axis=1)
                merged['group'] = merged.apply(lambda row: f"{row['label_text']}", axis=1)
                kurven_df['key_name'] = merged['key_name']
                kurven_df['group'] = merged['group']
                kurven_df = kurven_df.drop('key', axis=1)
                kurven_df = kurven_df.rename(columns={'key_name': 'key'})
                kurven_df['label_name'] = kurven_df['group'].map({'ok': 'OK-Schweißung', 
                                                                'near_ok_sonowechsel': 'NEAR-OK-Sonotrodenwechsel', 
                                                                'fehler_oel': 'Öl auf Terminalversatz', 
                                                                'fehler_leitungsversatz': 'Leitungsversatz', 
                                                                'fehler_terminversatz': 'Terminalversatz'})
                kurven_df = kurven_df.drop('group', axis=1)

                fig3 = gui_helper.class_curves(df, kurven, 'power')
                canvas3 = FigureCanvasTkAgg(fig3, master=frame_oben)
                canvas3.draw()
                canvas3.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=1)#, pady=(1,0))

            elif y_true:
                fig3 = gui_helper.zug_curves(df, kurven, 'power')
                canvas3 = FigureCanvasTkAgg(fig3, master=frame_oben)
                canvas3.draw()
                canvas3.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=1)#, pady=(1,0))

                #fig3 = gui_helper.plotIQRcurve(kurven_df, 'power')
                #canvas3 = FigureCanvasTkAgg(fig3, master=frame_rechts)
                #canvas3.draw()
                #canvas3.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=1, pady=(1,0))

                #fig4 = gui_helper.plotIQRcurve(kurven_df, 'force')
                #canvas4 = FigureCanvasTkAgg(fig4, master=frame_rechts)
                #canvas4.draw()
                #canvas4.get_tk_widget().grid(row=2, column=1, sticky="nsew", padx=1, pady=(1,0))

                #fig5 = gui_helper.plotIQRcurve(kurven_df, 'dist')
                #canvas5 = FigureCanvasTkAgg(fig5, master=frame_rechts)
                #canvas5.draw()
                #canvas5.get_tk_widget().grid(row=3, column=1, sticky="nsew", padx=1, pady=(1,0))
        else:
            messagebox.showerror("Fehler", f"Fehler bei der Verarbeitung der CSV-Datei.")


    ##############
    # MODELLTEST #
    ##############
    def run_model_test(self): 
        """ Modelltest ausführen 
            -> Klasse oder Zugfestigkeit muss gegeben sein
            -> Vorhersage des Modells mit der Realität vergleichen
        """
        # Alten Inhalt aus self.frame_show entfernen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

        self.frame_show.rowconfigure(0, weight=1)
        self.frame_show.rowconfigure(1, weight=9)
        self.frame_show.columnconfigure(0, weight=1)

        # frame_show in oben und unten aufteilen
        frame_oben = tk.Frame(self.frame_show, bg="white")
        frame_oben.grid(row=0, column=0, sticky="nsew")

        frame_unten = tk.Frame(self.frame_show, bg="white")
        frame_unten.grid(row=1, column=0, sticky="nsew")

        # Konfiguration für die Spalten innerhalb von frame_show
        frame_oben.grid_columnconfigure(0, weight=1)
        frame_oben.grid_columnconfigure(1, weight=2)
        frame_unten.grid_columnconfigure(0, weight=3)
        frame_unten.grid_columnconfigure(1, weight=1)

        # Überschrift
        headline = tk.Label(frame_oben, text="Modelltest: Vergleich der vorhergesagten und realen Daten", font=("Arial", 18, "bold"), 
                            bg="white", fg="black", anchor="w")
        headline.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        df, y_true, klasse_true = self.read_and_predict()

        # Informationen anzeigen
        if not y_true and not klasse_true: 
            label = tk.Label(frame_oben, text=f"➜ Es wurden keine gemessenen Zugfestigkeiten oder Klassifizierungen eingelesen.\n    Um einen Modelltest auszuführen, muss zu jeder Schweißkurve eine Zugfestigkeit und/oder eine Klasse vorliegen.", 
                              font=("Arial", 12), bg="white", fg="black", justify="left", anchor="w")
            label.grid(row=1, column=0, columnspan=2, sticky="w", padx=25)
            return
        if y_true and klasse_true:
            label = tk.Label(frame_oben, text=f"➜ Der Modelltest kann mit beiden Zielvariablen (Zugfestigkeit und Klasse) durchgeführt werden.", font=("Arial", 12), bg="white", fg="black", anchor="w")
            label.grid(row=1, column=0, columnspan=2, sticky="w", padx=25)
        else: 
            if y_true:
                ziel = "Zugfestigkeit" 
            else: 
                ziel = "Klasse"
            label = tk.Label(frame_oben, text=f"➜ Der Modelltest kann nur mit der {ziel} durchgeführt werden.", font=("Arial", 12), bg="white", fg="black", anchor="w")
            label.grid(row=1, column=0, columnspan=2, sticky="w", padx=25)

        ## Modelltest Klasse ##
        if klasse_true:
            # Konfusionsmatrix (oben links)
            label2 = tk.Label(frame_oben, text="Vorhersageergebnisse der Klassenzugehörigkeit", font=("Arial", 12, "bold"), bg="white", fg="black", anchor="w")
            label2.grid(row=2, column=0, columnspan=2, sticky="w", padx=25, pady=5)

            fig, ax = plt.subplots(figsize=(3, 3))
            cmatrix = confusion_matrix(df["klasse"].astype(int), df["klasse_pred"].astype(int)) # TODO newclass
            labels = np.unique(np.concatenate((df["klasse"].astype(int), df["klasse_pred"])))
            sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': "Anzahl der Schweißkurven"}, 
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title("Konfusionsmatrix", fontsize=10, weight="bold")
            ax.set_xlabel("Vorhergesagte Klassen", fontsize=9)
            ax.set_ylabel("Wahre Klasse", fontsize=9)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame_oben)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=3, column=0, sticky="nsew", padx=10)

            # Tabelle (oben rechts)
            columns = ("Versuchsnr.", "vorhergesagte Klasse", "Wahre Klasse")
            tree = ttk.Treeview(frame_oben, columns=columns, show="headings")
            style = ttk.Style()
            style.configure("Treeview", background="white", foreground="black", font=("Arial", 11))#, rowheight
            # Spaltenüberschriften
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=80)
            # Daten aus df in die Tabelle einfügen
            for _, row in df.iterrows():
                tree.insert("", tk.END, values=(int(row["nr"]), int(row["klasse_pred"]), int(row["klasse"])))
            tree.grid(row=3, column=1, sticky="nsew", padx=25, pady=10)
            # Scrollbar hinzufügen
            style.configure("Vertical.TScrollbar", troughcolor="white", background="white", bordercolor="darkgrey", arrowcolor="darkgrey")
            scrollbar = ttk.Scrollbar(frame_oben, orient="vertical", command=tree.yview, style="Vertical.TScrollbar")
            tree.configure(yscroll=scrollbar.set)
            scrollbar.grid(row=3, column=2, sticky="ns", padx=(0,100), pady=10)

            # Legende (rechts unter Konfusionsmatrix)
            legend_text = "\n".join([f"{i} → {label}" for i, label in enumerate(label_names)])
            legend_label = tk.Label(frame_oben, text=legend_text, font=("Arial", 10), fg="black", bg="white",
                                    justify="left", anchor="w", padx=10, pady=10, relief="solid", borderwidth=0) 
            legend_label.grid(row=4, column=0, sticky="nsew", padx=25)
        
            # SVC Metriken (rechts unter Tabelle)
            acc_test  = metrics.accuracy_score(df['klasse'], df['klasse_pred'])
            acc_label = tk.Label(frame_oben, text=(f"Genauigkeit des Modells auf den Testdaten = {acc_test}"), 
                                 font=("Arial", 11, "bold"), bg="white", fg="black", anchor="w", justify="left")
            acc_label.grid(row=4, column=1, sticky="nsew", padx=5)

            # 💾 Tabelle als CSV speichern
            prediction_path = os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "klassifikation.csv")
            df_result = pd.DataFrame({"Versuchsnr": df['nr'], "Klasse": df['klasse'], "Vorhersage": df['klasse_pred']})
            df_result.to_csv(prediction_path, sep=";", index=False, decimal=",")
            # Grafik als PNG speichern
            plt.savefig(os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "klassifikation_konfusionmatrix.png")) 
            

        ## Modelltest für Zugfestigkeit ##
        if y_true:
            # Linienplot (oben links)
            label3 = tk.Label(frame_unten, text="Vorhersageergebnisse der Zugfestigkeiten", font=("Arial", 12, "bold"), bg="white", fg="black", anchor="w")
            label3.grid(row=0, column=0, columnspan=2, sticky="w", padx=25, pady=5)

            # Vorhersage/Realität 
            if klasse_true:
                fig = gui_helper.plot_pred(df['y_pred'], df['zugfestigkeit'], df['klasse'], 'Regression der Testdaten', 'mit Linie', 1)
            else: 
                fig = gui_helper.plot_pred(df['y_pred'], df['zugfestigkeit'], None, 'Regression der Testdaten', 'mit Linie', 1)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame_unten)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=2, column=0, sticky="nsew", padx=10)

           # Tabelle (rechts)
            columns = ("Versuchsnr.", "vorhergesagte Zugfestigkeit", "Wahre Zugfestigkeit")
            tree2 = ttk.Treeview(frame_unten, columns=columns, show="headings")
            style2 = ttk.Style()
            style2.configure("Treeview", background="white", foreground="black", font=("Arial", 11))#, rowheight=35)
            # Spaltenüberschriften
            for col in columns:
                tree2.heading(col, text=col)
                tree2.column(col, anchor="center", width=100)
            # Daten aus df in die Tabelle einfügen
            for _, row in df.iterrows():
                tree2.insert("", tk.END, values=(int(row["nr"]), int(row["y_pred"]), int(row["zugfestigkeit"])))
            tree2.grid(row=2, column=1, sticky="nsew", padx=25, pady=10)
            # Scrollbar hinzufügen
            style.configure("Vertical.TScrollbar", troughcolor="white", background="white", bordercolor="darkgrey", arrowcolor="darkgrey")
            scrollbar = ttk.Scrollbar(frame_unten, orient="vertical", command=tree2.yview, style="Vertical.TScrollbar")
            tree2.configure(yscroll=scrollbar.set)
            scrollbar.grid(row=2, column=2, sticky="ns", padx=(0,100), pady=10)

            # SVR Metriken (rechts unter Tabelle)
            metrik_label = tk.Label(frame_unten, text=(f"Mittlerer absoluter Fehler (MAE) des Modells auf den Testdaten = {metrics.mean_absolute_error(df['zugfestigkeit'], df['y_pred']):.2f} MPa \n" 
                                    f"Bestimmtheitsmaß (R²) des Modells auf den Testdaten = {metrics.r2_score(df['zugfestigkeit'], df['y_pred']):.2f}"), 
                                    font=("Arial", 11, "bold"), bg="white", fg="black", anchor="w", justify="left")
            metrik_label.grid(row=3, column=1, sticky="nsew", padx=2)

            # 💾 Tabelle als CSV speichern
            prediction_path = os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "regression.csv")
            df_result = pd.DataFrame({"Versuchsnr": df['nr'], "Zugfestigkeit": df['zugfestigkeit'], "Vorhersage": df['y_pred']})
            # Grafik als PNG speichern
            plt.savefig(os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "regression_linineplot.png")) 
            df_result.to_csv(prediction_path, sep=";", index=False, decimal=",")

        return

    ####################
    # MODELLVORHERSAGE #
    ####################
    def run_prediction(self):
        """ Führt die Modellvorhersage """

        # Alten Inhalt aus self.frame_show entfernen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

        self.frame_show.rowconfigure(0, weight=1)
        self.frame_show.rowconfigure(1, weight=9)
        self.frame_show.columnconfigure(0, weight=1)

        # frame_show in oben und unten aufteilen
        frame_oben = tk.Frame(self.frame_show, bg="white")
        frame_oben.grid(row=0, column=0, sticky="nsew")

        frame_unten = tk.Frame(self.frame_show, bg="white")
        frame_unten.grid(row=1, column=0, sticky="nsew")

        # Konfiguration für die Spalten innerhalb von frame_show
        frame_oben.grid_columnconfigure(0, weight=1)
        frame_oben.grid_columnconfigure(1, weight=2)
        frame_unten.grid_columnconfigure(0, weight=1)
        frame_unten.grid_columnconfigure(1, weight=2)

        # Überschrift
        headline = tk.Label(frame_oben, text="Modellvorhersage", font=("Arial", 18, "bold"), 
                            bg="white", fg="black", anchor="w")
        headline.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        df, y_true, klasse_true = self.read_and_predict()

        # Informationen anzeigen
        if not os.path.exists(os.path.join(MODEL_DIR, self.machine_var.get(), "svc_stat.joblib")):
            label = tk.Label(frame_oben, text=f"➜ Es liegt kein Modell zur Vorhersage der Klassen vor.", font=("Arial", 12), bg="white", fg="black", anchor="w")
            label.grid(row=1, column=0, columnspan=2, sticky="w", padx=25)
        elif os.path.exists(os.path.join(MODEL_DIR, self.machine_var.get(), "svc_stat.joblib")) and os.path.exists(os.path.join(MODEL_DIR, self.machine_var.get(), "svr_stat.joblib")): 
            label = tk.Label(frame_oben, text=f"➜ Es liegen Modelle zur Vorhersage der Zugfestigkeiten und der Klassen vor.", 
                              font=("Arial", 12), bg="white", fg="black", justify="left", anchor="w")
            label.grid(row=1, column=0, columnspan=2, sticky="w", padx=25)

        ## Modellvorhersage Klasse ##
        if os.path.exists(os.path.join(MODEL_DIR, self.machine_var.get(), "svc_stat.joblib")):
            label2 = tk.Label(frame_oben, text="Vorhersageergebnisse der Klassenzugehörigkeit", font=("Arial", 12, "bold"), bg="white", fg="black", anchor="w")
            label2.grid(row=2, column=0, columnspan=2, sticky="w", padx=25, pady=5)
            # Barplot (oben links)
            fig, ax = plt.subplots(figsize=(3, 3))
            pd.Series(df['klasse_pred']).value_counts().sort_index().plot.bar(ax=ax, color="#003366")
            ax.set_title("Vorhersageverteilung", fontsize=10, weight="bold")
            ax.set_xlabel("Klassen", fontsize=9)
            ax.set_ylabel("Anzahl der Schweißversuche", fontsize=9)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame_oben)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=3, column=0, sticky="nsew", padx=10)

            # Tabelle (oben rechts)
            columns = ("Versuchsnr.", "vorhergesagte Klasse")
            tree = ttk.Treeview(frame_oben, columns=columns, show="headings")
            style = ttk.Style()
            style.configure("Treeview", background="white", foreground="black", font=("Arial", 11))#, rowheight
            # Spaltenüberschriften
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=80)
            # Daten aus df in die Tabelle einfügen
            for _, row in df.iterrows():
                tree.insert("", tk.END, values=(int(row["nr"]), int(row["klasse_pred"])))
            tree.grid(row=3, column=1, sticky="nsew", padx=25, pady=10)
            # Scrollbar hinzufügen
            style.configure("Vertical.TScrollbar", troughcolor="white", background="white", bordercolor="darkgrey", arrowcolor="darkgrey")
            scrollbar = ttk.Scrollbar(frame_oben, orient="vertical", command=tree.yview, style="Vertical.TScrollbar")
            tree.configure(yscroll=scrollbar.set)
            scrollbar.grid(row=3, column=2, sticky="ns", padx=(0,100), pady=10)

            if klasse_true:
                # Legende (rechts unter Konfusionsmatrix)
                legend_text = "\n".join([f"{i} → {label}" for i, label in enumerate(label_names)])
                legend_label = tk.Label(frame_oben, text=legend_text, font=("Arial", 10), fg="black", bg="white",
                                        justify="left", anchor="w", padx=10, pady=10, relief="solid", borderwidth=0) 
                legend_label.grid(row=4, column=0, sticky="nsew", padx=25)

            # 💾 Tabelle als CSV speichern
            prediction_path = os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "klassifikation.csv")
            df_result = pd.DataFrame({"Versuchsnr": df['nr'], "Vorhersage": df['klasse_pred']})
            df_result.to_csv(prediction_path, sep=";", index=False, decimal=",")
            # Grafik als PNG speichern
            plt.savefig(os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "klassifikation_balkengrafik.png")) 

        ## Modellvorhersage für Zugfestigkeit ##
        if os.path.exists(os.path.join(MODEL_DIR, self.machine_var.get(), "svr_stat.joblib")):
            label3 = tk.Label(frame_unten, text="Vorhersageergebnisse der Zugfestigkeiten", font=("Arial", 12, "bold"), bg="white", fg="black", anchor="w")
            label3.grid(row=0, column=0, columnspan=2, sticky="w", padx=25, pady=5)
      
            #if klasse_true:
            #    fig = gui_helper.plot_pred(df['y_pred'], df['zugfestigkeit'], df['klasse'], 'Regression der Testdaten', 'mit Linie', 1)
            #else: 
            #    fig = gui_helper.plot_pred(df['y_pred'], df['zugfestigkeit'], None, 'Regression der Testdaten', 'mit Linie', 1)

            # Histogramm
            fig, _ = plt.subplots(figsize=(3, 3))
            d = pd.DataFrame(df['y_pred'])
            d.columns=['zugf']
            d.zugf.plot(kind="hist", bins=20, color="#003366", edgecolor="black", alpha=1)
            plt.title("Histogramm der vorhergesagten Zugfestigkeiten", fontsize=10, weight="bold", pad=15)
            plt.xlabel("Zugfestigkeit in [MPa]", fontsize=9)
            plt.ylabel("Häufigkeit", fontsize=9)
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame_unten)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=2, column=0, sticky="nsew", padx=10)

            # Tabelle (rechts)
            columns = ("Versuchsnr.", "vorhergesagte Zugfestigkeit")
            tree2 = ttk.Treeview(frame_unten, columns=columns, show="headings")
            style2 = ttk.Style()
            style2.configure("Treeview", background="white", foreground="black", font=("Arial", 11))#, rowheight=35)
            # Spaltenüberschriften
            for col in columns:
                tree2.heading(col, text=col)
                tree2.column(col, anchor="center", width=100)
            # Daten aus df in die Tabelle einfügen
            for _, row in df.iterrows():
                tree2.insert("", tk.END, values=(int(row["nr"]), int(row["y_pred"])))
            tree2.grid(row=2, column=1, sticky="nsew", padx=25, pady=10)

            # TODO Tags für farbliche Markierungen definieren
            #tree2.tag_configure("good", background="darkgreen", foreground="white")
            #tree2.tag_configure("bad", background="darkred", foreground="white")

            # Daten aus df in die Tabelle einfügen mit farblicher Markierung
            #for _, row in df.iterrows():
            #    pred_value = int(row["y_pred"])
            #    tag = "good" if pred_value >= 2000 else "bad"
            #    tree2.insert("", tk.END, values=(int(row["nr"]), pred_value), tags=(tag,))

            # Scrollbar hinzufügen
            style.configure("Vertical.TScrollbar", troughcolor="white", background="white", bordercolor="darkgrey", arrowcolor="darkgrey")
            scrollbar = ttk.Scrollbar(frame_unten, orient="vertical", command=tree2.yview, style="Vertical.TScrollbar")
            tree2.configure(yscroll=scrollbar.set)
            scrollbar.grid(row=2, column=2, sticky="ns", padx=(0,100), pady=10)

            # 💾 Tabelle als CSV speichern
            prediction_path = os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "regression.csv")
            df_result = pd.DataFrame({"Versuchsnr": df['nr'], "Vorhersage": df['y_pred']})
            # Grafik als PNG speichern
            plt.savefig(os.path.join("GUI", self.machine_var.get(), "Ergebnisse", "regression_histogramm.png")) 
            df_result.to_csv(prediction_path, sep=";", index=False, decimal=",")


    #################
    # KURVENAUSWAHL #
    #################
    def select_curve(self):
        """ Auswahl einer bestimmten Schweißkurve """
        messagebox.showinfo("Info", "Button noch nicht implementiert.")

        # Alten Inhalt aus self.frame_show entfernen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

    ############
    #   SAVE   #
    ############
    def save_files(self):
        """ Speicherung der Daten """
        messagebox.showinfo("Info", "Button noch nicht implementiert. Speicherung der Tabelle und Grafiken erfolgt automatisch in den Ordner 'Vorhersageergebnisse'.")

        # Alten Inhalt aus self.frame_show entfernen
        for widget in self.frame_show.winfo_children():
            widget.destroy()

        """

        # Plot speichern oder anzeigen
    #plt.savefig(feature_txt.split()[0]+'.png', dpi=300, bbox_inches='tight')

        # Tabelle als CSV speichern
            prediction_path = os.path.join("GUI", machine, "Vorhersageergebnis", "regression.csv")
            df_result = pd.DataFrame({"Versuchsnr": versuche_test, "Vorhersage Zugfestigkeit": y_pred})
            # Plot oder Histogram als PNG speichern
            if y_true:
                df_result["Zugfestigkeit"] = y_true
                plt.savefig(os.path.join("GUI", machine, "Vorhersageergebnis", "regression_plot.png"))
            else:
                plt.savefig(os.path.join("GUI", machine, "Vorhersageergebnis", "regression_histogram.png")) 
            df_result.to_csv(prediction_path, sep=";", index=False, decimal=",")    
        """


def read_csv_skip_until_ms(file_path):
    """ Liest eine CSV-Datei ein und ignoriert alle Zeilen vor der Kopfzeile mit 'ms' """
    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if line.lower().startswith("ms"): 
                skip_rows = i
                break
    df = pd.read_csv(file_path, skiprows=skip_rows, sep=";", engine="python")
    df.columns = ["ms", "power", "force", "dist"] 
    return df


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
