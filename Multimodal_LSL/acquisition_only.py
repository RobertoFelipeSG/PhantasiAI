import time
import serial
import threading
from pyfirmata import Arduino, util

# 🔹 Définir le port série de l'Arduino (change si nécessaire)
PORT_ARDUINO = "/dev/ttyUSB0"  # Ex: COM5 sous Windows

# 🔹 Fichier où enregistrer les données
FILE_PATH = "/home/pi/PhantasiAI/Python/emg_data_2.1.txt"

# Initialisation globale
board = None
emg_pin = None
running = True  # Contrôle du thread


# 🔹 Connexion à l'Arduino
def try_connect_arduino(port=PORT_ARDUINO):
    global board, emg_pin
    try:
        board = Arduino(port)
        it = util.Iterator(board)
        it.start()

        # **Lire uniquement la broche A0**
        emg_pin = board.get_pin('a:0:i')
        print(" Arduino connecté, lecture sur A0.")
        return True
    except serial.SerialException:
        print(" Impossible de connecter l'Arduino.")
        return False


# 🔹 Lire la valeur du capteur EMG sur A0
def read_emg_data():
    sensor_value = emg_pin.read()
    return int(sensor_value * 1023) if sensor_value is not None else 0

# initaliser le fichier
def initialize_file():
    with open(FILE_PATH, "w") as file :
        file.write("EMG,Timestamp\n")

# 🔹 Fonction pour enregistrer les données
def log_data(timestamp, emg):
    with open(FILE_PATH, "a") as file:
        file.write(f"{emg},{timestamp:.2f}\n")


# 🔹 Thread pour l'acquisition en continu
def data_acquisition_thread():
    global running
    timestamp = 0

    print("📡 Acquisition des données EMG en cours...")

    while running:
        start_time = time.perf_counter()

        # **Lire et enregistrer la donnée**
        emg_value = read_emg_data()
        log_data(timestamp, emg_value)

        #print(f" {timestamp:.2f} sec -> EMG: {emg_value}")

        timestamp += 0.05  # **20 Hz = une mesure toutes les 0.05 secondes**

        # Pause pour respecter la fréquence d'échantillonnage
        elapsed_time = time.perf_counter() - start_time
        time.sleep(max(0.05 - elapsed_time, 0))


# 🔹 Fonction pour arrêter proprement
def stop_acquisition():
    global running
    running = False
    print("\n Arrêt de l'acquisition.")
    if board is not None:
        board.exit()


# 🔹 Main : démarrer l'acquisition en arrière-plan
if __name__ == "__main__":
    initialize_file()
    is_arduino_connected = try_connect_arduino()

    if is_arduino_connected:
        acquisition_thread = threading.Thread(target=data_acquisition_thread, daemon=True)
        acquisition_thread.start()

        try:
            while True:
                time.sleep(1)  # Le programme continue à tourner
        except KeyboardInterrupt:
            stop_acquisition()
