import numpy as np
import pandas as pd
import time
from scipy.signal import welch

# ---- Fonctions de calcul des caractéristiques EMG ----
def mean_absolute_value(signal):
    return np.mean(np.abs(signal))

def root_mean_square(signal):
    return np.sqrt(np.mean(np.square(signal)))

def variance(signal):
    return np.var(signal)

def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def slope_sign_changes(signal, threshold=0):
    return np.sum(((signal[:-2] - signal[1:-1]) * (signal[1:-1] - signal[2:])) > threshold)

def frequency_features(signal, fs=20):
    """Calcul des fréquences moyenne et médiane"""
    f, Pxx = welch(signal, fs=fs, nperseg=len(signal)//2)
    mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
    median_freq = f[np.where(np.cumsum(Pxx) >= np.sum(Pxx) / 2)[0][0]]
    return mean_freq, median_freq

# ---- Extraction des caractéristiques sur les sous-parties ----
def extract_emg_features_per_segment(signal, num_segments=10, fs=20):
    """Divise le signal en `num_segments` parties et calcule les caractéristiques sur chaque partie"""
    signal = np.array(signal)
    segment_length = len(signal) // num_segments
    features_per_segment = {key: [] for key in ["MAV", "RMS", "Variance", "WL", "SSC", "MeanFreq", "MedianFreq"]}

    for i in range(num_segments):
        segment = signal[i * segment_length:(i + 1) * segment_length]
        features_per_segment["MAV"].append(mean_absolute_value(segment))
        features_per_segment["RMS"].append(root_mean_square(segment))
        features_per_segment["Variance"].append(variance(segment))
        features_per_segment["WL"].append(waveform_length(segment))
        features_per_segment["SSC"].append(slope_sign_changes(segment))
        mean_freq, median_freq = frequency_features(segment, fs)
        features_per_segment["MeanFreq"].append(mean_freq)
        features_per_segment["MedianFreq"].append(median_freq)

    return features_per_segment

# ---- Fonction pour extraire la dernière minute ----
def load_last_minute_emg_data(file_path, selected_features, num_segments=10, fs=20):
    """Charge la dernière minute de données EMG et extrait les caractéristiques."""
    data = pd.read_csv(file_path, sep=",", header=0, names=["EMG", "Timestamp"])
    data["minute"] = (data["Timestamp"] // 60).astype(int)

    # Obtenez la dernière minute
    last_minute = data["minute"].max()
    last_minute_data = data[data["minute"] == last_minute]["EMG"].values
    print
    
    print(last_minute_data[-1])

    # Extraction des caractéristiques sur chaque segment de la dernière minute
    segmented_features = extract_emg_features_per_segment(last_minute_data, num_segments, fs)

    # Filtrage des caractéristiques sélectionnées
    filtered_features = {key: segmented_features[key] for key in selected_features}

    return filtered_features

# ---- Fonction d'enregistrement des données dans un fichier ----
def save_subject_in_file(file, vector, n_suj, n_dim, n_val, interaction_scale, interaction_sign):
    """Sauvegarde les données de chaque sujet dans le fichier"""
    file.write(str(n_suj) + ";")
    for i_dim in range(n_dim):
        for i_val in range(n_val):
            # Arrondi des valeurs à 2 décimales et garde-les sous forme de float
            file.write(f"{round(vector[i_dim][i_val], 2)} ")
        file.write(";")
    file.write(f"{interaction_scale};{interaction_sign}\n")

def save_data_in_file(vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale):
    """Sauvegarde les données dans un fichier de sortie"""
    with open("/home/pi/PhantasiAI/Python/subject_3dim.txt", "w") as file:
        file.write("Sujet;")
        for i_dim in range(n_dim):
            file.write(titles[i_dim] + ";")  # Utilise les titres des features sélectionnées
        file.write("Interaction_Scale;Interaction_Sign\n")
        for i_suj in range(n_suj):
            save_subject_in_file(file, vector[i_suj], i_suj, n_dim, n_val, interaction_scale, interaction_sign)

# ---- Programme principal ----
if __name__ == "__main__":
    selected_features = ["MAV", "RMS"]  # Définissez les caractéristiques que vous souhaitez ici
    while True:
        # Définir les paramètres de la simulation
        n_suj = 1
        n_dim = len(selected_features)  # Nombre de dimensions (sélectionnées)
        n_val = 10  # Nombre de valeurs par dimension (segments)
        interaction_sign = 1
        interaction_scale = 0.5
        file_path = "/home/pi/PhantasiAI/Python/emg_data_2.1.txt"  # Fichier source
    
        
        titles = selected_features  # Utilise les features sélectionnées comme titres

        # Charger les données EMG et extraire la dernière minute
        features_per_minute = load_last_minute_emg_data(file_path, selected_features, num_segments=n_val)

        # Vérifier qu'il y a des caractéristiques extraites
        if features_per_minute:
            # Créer le vecteur des caractéristiques à partir de la dernière minute
            vector = [list(features_per_minute.values())]

            # Affichage des résultats avant d'enregistrer
            print(f"New vector data: {[[round(float(val),2) for val in sublist] for sublist in vector[0]]}")

            # Sauvegarder les données dans le fichier de sortie
            save_data_in_file(vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale)
            print("File saved successfully!")

        time.sleep(50)  # Rafraîchir toutes les 50 secondes