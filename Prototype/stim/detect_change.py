import os
from pathlib import Path
import time
import subprocess
from watchdog.observers import Observer
#from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import paho.mqtt.client as mqtt
#from paho.mqtt.client import CallbackAPIVersion


# Find current base directory and  thus find the results from the session
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
session_dir = os.path.join(base_dir, "emg", "emg-recordings")#, timestamp)
print(session_dir)

# Finding the last directory based on a timestamp
all_subdirs = [os.path.join(session_dir, d) for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
if all_subdirs:
    latest_subdir = max(all_subdirs, key=os.path.getctime)
    os.chdir(latest_subdir)
    print(f"Navigated to the last created folder: {os.getcwd()}")
else:
    print("No subdirectories found in the specified parent directory.")
    

class ChangeHandler(FileSystemEventHandler):

    def __init__(self, script_path, mqtt_client):
        self.script_path = script_path
        self.mqtt_client = mqtt_client  # Référence au client MQTT

    def on_modified(self, event):
        if event.src_path.endswith("peak_classificat.txt"):
            print(f"Changement détecté dans {event.src_path}. Exécution du script GPBO...")
            self.run_script()
            self.mqtt_client.publish("start", "on")
            print("Message 'on' publié sur le topic 'start'.")

    def run_script(self):
        try:
            subprocess.run([
                "python3", self.script_path,
                "--n_iters", "20",
                "--n_rnd", "1",
                "--kappa", "3.0",
                "--AF_name", "EI",
                "--noise_level", "0.1"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script : {e}")

def on_connect(client, userdata, flags, rc):
    print(f"Connecté avec le code {rc}")

if __name__ == "__main__":
    
   
    script_to_run = os.path.join(base_dir, "stim", "gp_code.py") #"/home/phantasiai/Prototype/4th_prot_LDA_xgboost/stim/gp_code.py"
    print(script_to_run)
    #path_to_watch = "/home/pi/PhantasiAI/Python/"  # Répertoire contenant 'subject_3dim.txt'
    #script_to_run = "/home/pi/PhantasiAI/Python/gp_code.py"

    # Configuration du client MQTT
    broker_address = "localhost"
    mqtt_client = mqtt.Client("FileWatcher_Client")
    #mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1, "Filewatcher_Client")
    mqtt_client.on_connect = on_connect

    # Connexion au broker MQTT
    mqtt_client.connect(broker_address, 1883, 60)
    mqtt_client.loop_start()

    # Créer le gestionnaire de changements et observer les modifications dans le fichier
    event_handler = ChangeHandler(script_to_run, mqtt_client)
    
    # Créer un observateur qui surveille les modifications du répertoire spécifié
    observer = Observer()
    observer.schedule(event_handler, path=latest_subdir, recursive=False)
    
    print(f"Surveillance des changements dans {latest_subdir}...")#path_to_watch}...")
    observer.start()

    # Boucle pour garder le programme actif et surveiller les changements
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt de la surveillance.")
        observer.stop()

    # Arrêter le client MQTT et l'observateur proprement
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    observer.join()
