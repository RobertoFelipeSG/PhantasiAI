import hashlib
import time
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.connection_manager import logging

class ChangeHandler(FileSystemEventHandler):
    '''
    Handles change detection from WatchDog observing
    Runs GPBO script if change in content is detected in features.txt file
    '''

    def __init__(self, profiler, mqtt_client, client_topic, optimizer):
        self.profiler = profiler
        self.mqtt_client = mqtt_client
        self.topic = client_topic
        self.optimizer = optimizer
        
        self.last_hash = None
        self.last_trial = None
        self.is_running = False
        self.last_runtime = 0
        self.debounce_interval = 0.5 # prevents debounce checks (1s after optimization starts and 1s after stimulation ends

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
        if event.is_directory or not event.src_path.endswith("features.txt"): 
            return
        
        # Ensure previous optimization is complete
        if self.is_running:
            logging.info(f"[WatchDog] Change detected but previous optimization still running. Skipping GPBO...")
            return
        
        # Debounce check
        current_time = time.time()
        if (current_time - self.last_runtime) < self.debounce_interval:
            logging.info(f"[WatchDog] Change detected but within debounce interval. Skipping GPBO...")
            return

        # Check content of new features file
        new_hash = self._get_file_hash(event.src_path)
        curr_trial = self.profiler.current_trial

        if new_hash is None: # Safely ignore momentary 0-byte file truncation event
            return

        # skip if hash identical AND new trial
        if curr_trial == self.last_trial and new_hash == self.last_hash:
            logging.info(f"[WatchDog] Change detected but identical content. Skipping GPBO...")
            return

        # Run GPBO script
        try:
            # State updates (if content is new)
            self.is_running = True
            self.last_trial = curr_trial # store current trial for future validation checks
            logging.info(f"[WatchDog] Valid change detected in {event.src_path} for trial {curr_trial}. Triggering GPBO...")
            
            self.last_hash = new_hash
            self.last_runtime = current_time
        
            self.mqtt_client.publish(f"{self.topic}/GPBO/start", "on")
            
            self.optimizer.run(event.src_path, curr_trial)
        except ValueError as e:
            logging.error(f"[WatchDog] GPBO aborted due to bad data: {e}")
        except Exception as e:
            logging.error(f"[WatchDog] Error running GPBO: {e}", exc_info=True)
        finally:
            self.last_hash = self._get_file_hash(event.src_path) # ensures last file hash represents full features.txt file
            self.is_running = False # ensures new optimization only occurs once current + stimulation is complete

class WatchDogOpt:
    '''
    Detect change in specified folder for optimization handling
    '''
    def __init__(self, profiler, mqtt_client, client_topic, optimizer):
        self.observer = None
        self.profiler = profiler
        self.mqtt_client = mqtt_client
        self.topic = client_topic
        self.optimizer = optimizer
    
    def start_watching(self, directory_path):
        if self.observer: self.stop_watching()

        self.observer = Observer()
        handler = ChangeHandler(self.profiler, self.mqtt_client, self.topic, self.optimizer)
        self.observer.schedule(handler, path=directory_path, recursive=False)
        self.observer.start()

        logging.info(f"[WatchDog] initialized for change detection in {directory_path}")

    def stop_watching(self): 
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
