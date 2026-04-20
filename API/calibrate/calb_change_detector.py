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

    def __init__(self, profiler, calibrator):
        self.profiler = profiler
        self.calibrator = calibrator
        
        self.last_hash = None
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
        
        # Ensure previous stimulation is complete
        if self.is_running:
            logging.info(f"[WatchDog] Change detected but previous stimulation still running. Skipping stimulation...")
            return
        
        # Debounce check
        current_time = time.time()
        if (current_time - self.last_runtime) < self.debounce_interval:
            logging.info(f"[WatchDog] Change detected but within debounce interval. Skipping stimulation...")
            return

        # Check content of new features file
        new_hash = self._get_file_hash(event.src_path)
        if new_hash == self.last_hash:
            logging.info(f"[WatchDog] Change detected but identical content. Skipping stimulation...")
            return

        # Run calibration script
        try:
            # State updates (if content is new)
            self.is_running = True
            curr_trial = self.profiler.current_trial # get the current trial as soon as valid change detected
            logging.info(f"[WatchDog] Valid change detected in {event.src_path} for trial {curr_trial}. Triggering stimulation...")
            
            self.last_hash = new_hash
            self.last_runtime = current_time
            
            self.calibrator.run(event.src_path, curr_trial)
        except ValueError as e:
            logging.error(f"[WatchDog] Calibration aborted due to bad data: {e}")
        except Exception as e:
            logging.error(f"[WatchDog] Error running calibrator: {e}", exc_info=True)
        finally:
            self.last_hash = self._get_file_hash(event.src_path) # ensures last file hash represents full features.txt file
            self.is_running = False # ensures new optimization only occurs once current + stimulation is complete

class WatchDogCalb:
    '''
    Detect change in specified folder for optimization handling
    '''
    def __init__(self, profiler, calibrator):
        self.profiler = profiler
        self.observer = None
        self.calibrator = calibrator
    
    def start_watching(self, directory_path):
        if self.observer: self.stop_watching()

        self.observer = Observer()
        handler = ChangeHandler(self.profiler, self.calibrator)
        self.observer.schedule(handler, path=directory_path, recursive=False)
        self.observer.start()

        logging.info(f"[WatchDog] initialized for change detection in {directory_path}")

    def stop_watching(self): 
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None