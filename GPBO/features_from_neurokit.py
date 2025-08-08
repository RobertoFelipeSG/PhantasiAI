import numpy as np
import pandas as pd
import time
import neurokit2 as nk
import matplotlib.pyplot as plt
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

def frequency_features(signal, fs=220):
    """Calcul des fréquences moyenne et médiane"""
    f, Pxx = welch(signal, fs=fs, nperseg=len(signal)//2)
    mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
    median_freq = f[np.where(np.cumsum(Pxx) >= np.sum(Pxx) / 2)[0][0]]
    return mean_freq, median_freq

# ---- Extraction des caractéristiques sur les sous-parties ----
def extract_emg_features_per_segment(signal, num_segments=10, fs=220):
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

# ---- Fonction pour extraire la dernière minute et traiter les activations musculaires ----
def load_and_process_emg_data(file_path, threshold=0.1, num_segments=10, fs=220):
    """Charge la dernière minute de données EMG, traite le signal et extrait les caractéristiques."""
    # Charger les données
    data = pd.read_csv(file_path, sep=",", header=0, names=["EMG", "Timestamp"])
    data["minute"] = (data["Timestamp"] // 60).astype(int)

    # Dernière minute
    last_minute = data["minute"].max()
    last_minute_data = data[data["minute"] == last_minute]["EMG"].values


    

    # Prétraitement avec NeuroKit2
    emg_signals, info = nk.emg_process(last_minute_data, sampling_rate=fs)

    # Détecter les activations musculaires : seuil sur l'amplitude
    activation = emg_signals["EMG_Clean"] > threshold * max(emg_signals["EMG_Clean"])

    # Extraction des caractéristiques sur chaque segment de la dernière minute
    segmented_features = extract_emg_features_per_segment(emg_signals["EMG_Clean"], num_segments, fs)

    return emg_signals, activation, segmented_features, info

# ---- Fonction d'enregistrement des données dans un fichier ----
def save_data_in_file(vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale):
    """Sauvegarde les données dans un fichier de sortie"""
    with open("subject_3dim.txt", "w") as file:
    # with open("/home/pi/PhantasiAI/Python/subject_3dim.txt", "w") as file:
        file.write("Sujet;")
        for i_dim in range(n_dim):
            file.write(titles[i_dim] + ";")  # Utilise les features sélectionnées comme titres
        file.write("Interaction_Scale;Interaction_Sign\n")
        for i_suj in range(n_suj):
            file.write(f"{n_suj};")
            for i_dim in range(n_dim):
                for i_val in range(n_val):
                    file.write(f"{round(vector[i_suj][i_dim][i_val], 2)} ")
                file.write(";")
            file.write(f"{interaction_scale};{interaction_sign}\n")


# ---- Fonction principale ----
if __name__ == "__main__":
    selected_features = ["MAV", "RMS"]
    num_segments = 10
    sampling_rate = 220
    threshold = 0.1
    file_path = "emg_data_2.1.txt"
    interaction_sign = 1
    interaction_scale = 0.5

    n_suj = 1
    n_dim = len(selected_features)
    n_val = num_segments
    titles = selected_features  # Utilise les features sélectionnées comme titres

    while True:
        # Charger, traiter et extraire les caractéristiques
        emg_signals, activation, segmented_features, info = load_and_process_emg_data(file_path, threshold, num_segments, sampling_rate)

        # Créer le vecteur des caractéristiques à partir de la dernière minute
        vector = [list(segmented_features.values())]

        # Affichage des résultats avant d'enregistrer
        print(f"New vector data: {[[round(float(val),2) for val in sublist] for sublist in vector[0]]}")

        # Sauvegarder les données dans le fichier de sortie
        save_data_in_file(vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale)
        print("File saved successfully!")

        # Visualisation du signal EMG prétraité et des activations
        nk.emg_plot(emg_signals, info)
        plt.show()

        time.sleep(60)  # Rafraîchir toutes les 50 secondes
