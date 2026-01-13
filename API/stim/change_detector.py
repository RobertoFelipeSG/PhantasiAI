from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.connection_manager import logging
from stim.gpbo import GPBOptimizer

class ClassificationChangeHandler(FileSystemEventHandler):
    '''
    Handles change detection from WatchDog observing
    Runs GPBO script if change detected in current peak_classification.txt file 
    '''

    def __init__(self, mqtt_client):
        self.mqtt_client = mqtt_client 

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith("peak_classification.txt"):
            logging.info(f"Change detected in {event.src_path}. Triggering GPBO...")
        
            self.mqtt_client.publish("start", "on")

            # Run GPBO script
            try:
                optimizer = GPBOptimizer(mqtt_client=self.mqtt_client, subject_file_path=event.src_path)
                optimizer.run()
            except Exception as e:
                logging.error(f"Error running GPBO: {e}", exc_info=True)

class WatchDog:
    '''
    Detect change in classification folder that contains peak_classification.txt file
    '''
    def __init__(self, mqtt_client):
        self.observer = None
        self.mqtt_client = mqtt_client
    
    def start_watching(self, directory_path):
        if self.observer: self.stop_watching()

        self.observer = Observer()
        handler = ClassificationChangeHandler(self.mqtt_client)
        self.observer.schedule(handler, path=directory_path, recursive=False)
        self.observer.start()

        logging.info(f"WatchDog initialized for change detection in {directory_path}")

    def stop_watching(self): 
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
