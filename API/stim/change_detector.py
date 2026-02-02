import hashlib
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.connection_manager import logging
from stim.gpbo import GPBOptimizer
from stim.square import ParamGenerator

class ChangeHandler(FileSystemEventHandler):
    '''
    Handles change detection from WatchDog observing
    Runs GPBO script if change in content is detected in peak_classification.txt file
    Runs stimulation param generator script if change in content is detected in stim.txt file
    '''

    def __init__(self, mqtt_client, client_topic, change_type):
        self.mqtt_client = mqtt_client
        self.topic = client_topic
        self.change_type = change_type
        self.last_hash = None
        self.last_runtime = 0
        self.debounce_interval = 1.0 # prevents debounce checks

    def _get_file_hash(self, filepath): 
        '''
        Reads file and returns unique MD5 hash of content
        Returns None if file is empty or locked
        '''
        hasher = hashlib.md5()
        
        try:
            with open(filepath, 'rb') as f:
                buf = f.read()
                if not buf: return None # empty file
                hasher.update(buf)
            return hasher.hexdigest()
        except (IOError, OSError):
            return None

    def on_modified(self, event):
        if self.change_type == 'features':
            if event.is_directory or not event.src_path.endswith("peak_features.txt"): 
                return
            
            # Debounce check
            current_time = time.time()
            if (current_time - self.last_runtime) < self.debounce_interval:
                return
    
            # Check content of new classification file
            new_hash = self._get_file_hash(event.src_path)
            if new_hash == self.last_hash:
                logging.info(f"[WatchDog] New file detected but identical content. Skipping GPBO...")
                return
            
            # State updates (if content is new)
            logging.info(f"[WatchDog] Change detected in {event.src_path}. Triggering GPBO...")
            self.last_hash = new_hash
            self.last_runtime = current_time
        
            self.mqtt_client.publish(f"{self.topic}/GPBO/start", "on")
    
            # Run GPBO script
            try:
                optimizer = GPBOptimizer(file_path=event.src_path, 
                                         mqtt_client=self.mqtt_client, 
                                         client_topic=self.topic)
                optimizer.run()
            except ValueError as e:
                logging.error(f"[WatchDog] GPBO aborted due to bad data: {e}")
            except Exception as e:
                logging.error(f"[WatchDog] Error running GPBO: {e}", exc_info=True)

        elif self.change_type == 'stimulation':
            if event.is_directory or not event.src_path.endswith("stim.txt"): 
                return
            
            # Debounce check
            current_time = time.time()
            if (current_time - self.last_runtime) < self.debounce_interval:
                return
    
            # Check content of new classification file
            new_hash = self._get_file_hash(event.src_path)
            if new_hash == self.last_hash:
                logging.info(f"[WatchDog] New file detected but identical content. Skipping param generation...")
                return
            
            # State updates (if content is new)
            logging.info(f"[WatchDog] Change detected in {event.src_path}. Triggering param generation...")
            self.last_hash = new_hash
            self.last_runtime = current_time
        
            self.mqtt_client.publish(f"{self.topic}/Square/start", "on")
    
            # Run param generator script
            try:
                param_generator = ParamGenerator(file_path=event.src_path, 
                                                 mqtt_client=self.mqtt_client,
                                                 client_topic=self.topic)
                param_generator.run()
            except ValueError as e:
                logging.error(f"[WatchDog] Param generation aborted due to bad data: {e}")
            except Exception as e:
                logging.error(f"[WatchDog] Error running param generation: {e}", exc_info=True)

class WatchDog:
    '''
    Detect change in specified folder
    '''
    def __init__(self, mqtt_client, client_topic, change_type):
        self.observer = None
        self.mqtt_client = mqtt_client
        self.topic = client_topic
        self.change_type = change_type
    
    def start_watching(self, directory_path):
        if self.observer: self.stop_watching()

        self.observer = Observer()
        handler = ChangeHandler(self.mqtt_client, self.topic, self.change_type)
        self.observer.schedule(handler, path=directory_path, recursive=False)
        self.observer.start()

        logging.info(f"[WatchDog] initialized for change detection in {directory_path}")

    def stop_watching(self): 
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
