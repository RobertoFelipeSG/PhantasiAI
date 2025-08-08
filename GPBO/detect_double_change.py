import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import paho.mqtt.client as mqtt

class ChangeHandler(FileSystemEventHandler):

    def __init__(self, script_path, mqtt_client, path):
        self.script_path = script_path
        self.mqtt_client = mqtt_client  # Référence au client MQTT
        self.path = path

    def on_modified(self, event):
        if event.src_path.endswith(self.path):
            print(f"Changement détecté dans {event.src_path}. Exécution du script ")
            self.run_script()
            self.mqtt_client.publish("start", "on")
            print("Message 'on' publié sur le topic 'start'.")
        
    def run_script(self):
        try:
            subprocess.run([
                "python3", self.script_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script : {e}")

def on_connect(client, userdata, flags, rc):
    print(f"Connecté avec le code {rc}")

if __name__ == "__main__":

    path_to_watch = "/home/pi/PhantasiAI/Python/"  # Répertoire contenant 'subject_3dim.txt'
    script_to_run_1 = "/home/pi/PhantasiAI/Python/gp_code.py"
    script_to_run_2 = "/home/pi/PhantasiAI/Python/Square.py"


    # Configuration du client MQTT
    broker_address = "localhost"
    mqtt_client = mqtt.Client("FileWatcher_Client")
    mqtt_client.on_connect = on_connect

    # Connexion au broker MQTT
    mqtt_client.connect(broker_address, 1883, 60)
    mqtt_client.loop_start()

    # Créer le gestionnaire de changements et observer les modifications dans le fichier
    event_handler_1 = ChangeHandler(script_to_run_1, mqtt_client, "hist_params.npy")
    event_handler_2 = ChangeHandler(script_to_run_2, mqtt_client, "subject_3dim.txt")

    # Créer un observateur qui surveille les modifications du répertoire spécifié
    observer_1 = Observer()
    observer_1.schedule(event_handler_1, path=path_to_watch, recursive=False)
    observer_2 = Observer()
    observer_2.schedule(event_handler_2, path=path_to_watch, recursive=False)

    print(f"Surveillance des changements dans {path_to_watch}...")
    observer_1.start()
    observer_2.start()

    # Boucle pour garder le programme actif et surveiller les changements
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt de la surveillance.")
        observer_1.stop()
        observer_2.stop()

    # Arrêter le client MQTT et l'observateur proprement
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    observer_1.join()
    observer_2.join()