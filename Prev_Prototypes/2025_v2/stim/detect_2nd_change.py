import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import paho.mqtt.client as mqtt
import os

from Previous_Prototypes.Prototype.stim.detect_change import latest_subdir

class ChangeHandler(FileSystemEventHandler):

    def __init__(self, script_path, mqtt_client):
        self.script_path = script_path
        print(self.script_path)
        self.mqtt_client = mqtt_client  # Référence au client MQTT

    def on_modified(self, event):
        if event.src_path.endswith("stim.txt"):
            print(f"Changement détecté dans {event.src_path}. Exécution du script Square...")
            self.run_script()
            self.mqtt_client.publish("start", "on")
            print("Message 'on' publié sur le topic 'start'.")

    def run_script(self):
        try:
            subprocess.run([
                "python3", self.script_path,
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script : {e}")

def on_connect(client, userdata, flags, rc):
    print(f"Connecté avec le code {rc}")

if __name__ == "__main__":

    #path_to_watch = os.getcwd()
    path_to_watch = latest_subdir
    #print('here', path_to_watch)
    script_to_run = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Square.py'))
    #path_to_watch = "/home/pi/PhantasiAI/Python/"  # Répertoire contenant 'subject_3dim.txt'
    #script_to_run = "/home/pi/PhantasiAI/Python/Square.py"

    # Configuration du client MQTT
    broker_address = "localhost"
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "SQWatcher_Client")
    mqtt_client.on_connect = on_connect

    # Connexion au broker MQTT
    mqtt_client.connect(broker_address, 1883, 60)
    mqtt_client.loop_start()

    # Créer le gestionnaire de changements et observer les modifications dans le fichier
    event_handler = ChangeHandler(script_to_run, mqtt_client)

    # Créer un observateur qui surveille les modifications du répertoire spécifié
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)

    print(f"Surveillance des changements dans {path_to_watch}...")
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
