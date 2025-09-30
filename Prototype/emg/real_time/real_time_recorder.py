
import time
import csv
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from sibling directories
from utils.path_utils import add_parent_to_syspath
add_parent_to_syspath(1) 

from ..offline_analysis.peak_detector import PeakDetector
from ..offline_analysis.peak_classifier import PeakClassifier



class RealTimeRecorder:
    def __init__(self, parent):
        self.parent = parent
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.event_times = []
        self.filename = None
        self.session_dir = None
        self._index = 0
        self.channel_count = 0
        self.last_event_time = 0 # track the last event time for automatic marking
        self._marked_events = set() # track which events had been written to the csv
        print(f"[Recorder] initialized with parent: {parent}")

    def start_recording(self):
        if self.recording:
            return

        timestamp = time.strftime("%Y-%m-%d_%Hh%Mm%S", time.localtime())
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.session_dir = os.path.join(base_dir, "emg-recordings", timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        self.filename = os.path.join(self.session_dir, "emg.csv")

        try:
            self.csv_file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)

            self.channel_count = getattr(self.parent.graph, 'num_channels', 1)
            header = ['timestamp'] + [f'ch{i+1} (µV)' for i in range(self.channel_count)] + ['event']
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.recording = True
            self._marked_events.clear() # reset marked events for new recording 
            print(f"[Recorder] Recording started: {os.path.basename(self.filename)}")
            self.parent.chat.log_event(f"[Recorder] Recording started: {os.path.basename(self.filename)}")
        except Exception as e:
            print(f"[Recorder] Failed to start recording: {e}")
            self.recording = False

    def record_data_point(self, timestamp, emg_values):
        if not self.recording or not self.csv_writer:
            return

        try:
            if isinstance(emg_values, (np.ndarray, list)):
                emg_values = [float(v) for v in emg_values]
            else:
                emg_values = [float(emg_values)]
            
            # check if current timestamp is within 10ms of any event time
            event_flag = any(abs(timestamp - t) < 0.01 for t in self.event_times)        
            
            # only set flag if we haven't already marked the event 
            if event_flag:
                # check if we've already marked this event 
                event_already_marked = False
                
                for event_time in self.event_times:
                    # check if we've already written 1
                    if abs(timestamp - event_time) < 0.01:
                        if hasattr(self,'_marked_events') and event_time in self._marked_events:
                            event_flag = False
                            break
                        else:
                            # mark this event as processed
                            if not hasattr(self,'_marked_events') and event_time in self._marked_events:
                                self._marked_events = set()
                            self._marked_events.add(event_time)
                            print(f"[Recorder] Event marked at timestamp {timestamp:.3f}s for event at {event_time:.3f}s")
                            break
                            
                
                
            #if len(self.event_times) > 0 and self._index % 100 == 0: # log every 100th datapoint
                #print(f"[Recorder] Current timestamp: {timestamp:.3f}s, event times: {[f'{t:.3f}' for t in self.event_times[-5:]]}, last event time: {self.last_event_time:.3f}s")
            
            emg_values = emg_values[:self.channel_count]
            row = [f"{timestamp:.4f}"] + [f"{v:.2f}" for v in emg_values] + [int(event_flag)]
            self.csv_writer.writerow(row)

            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1
        except Exception as e:
            print(f"[Recorder] Failed to record data point: {e}")

    def stop_recording(self):
        if not self.recording:
            return
        try:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.csv_writer = None
            self.recording = False
            self.parent.chat.log_event("Recording stopped")

            #if self.filename:
                # Use the XGBoost peak classifier for classification
            #    analyzer = PeakClassifier(csv_path=self.filename)
            #    results = analyzer.run(show_plots=False, save_results=True, classify_peaks=True)
                
                # Log classification results if available
            #    if results.get('classifications'):
            #        num_classifications = len(results['classifications'])
            #        self.parent.chat.log_event(f"Peak analysis completed: {results['num_peaks']} peaks detected, {num_classifications} classified")
                    
                    # Show classification summary
            #        class_counts = {}
            #        for result in results['classifications']:
            #            cls = result['predicted_class']
            #            class_counts[cls] = class_counts.get(cls, 0) + 1
                    
            #        summary = ", ".join([f"{cls}% MVC: {count}" for cls, count in sorted(class_counts.items())])
            #        self.parent.chat.log_event(f"Classification summary: {summary}")
            #    else:
            #        self.parent.chat.log_event(f"Peak analysis completed: {results['num_peaks']} peaks detected (no classification available)")

        except Exception as e:
            print(f"[Recorder] Failed to stop recording: {e}")

    def mark_event(self, timestamp = None): # Related to M manual keypress
        if not self.recording:
            return
        try:
            if timestamp is not None:
                # use the provided timestamp from data recording
                current_time = timestamp
            elif hasattr(self.parent, 'graph') and hasattr(self.parent.graph, 'thread'):
                # use synchronized timing from data thread
                current_time = self.parent.graph.thread.get_current_time()
            else:
                # last resort fallback
                current_time = time.time()

            self.event_times.append(current_time)
            self.last_event_time = current_time # update the last event time
            #print(f"[Recorder] Event marked at {current_time:.3f}s, total events: {len(self.event_times)}, last event time: {self.last_event_time:.3f}s")
            self.parent.chat.log_event(f"Event marked at {current_time:.3f}s")
        except Exception as e:
            print(f"[Recorder] Failed to mark event: {e}")
    
    def mark_event_with_timestamp(self, timestamp = None): # Mark an event with a specific timestamp (for automatic markers)
        if not self.recording:
            return
        try:
            self.event_times.append(timestamp)
            self.last_event_time = timestamp
            #print(f"[Recorder] Event marked with timestamp {timestamp:.3f}s, total events: {len(self.event_times)}")
            self.parent.chat.log_event(f"Automatic marker event at {timestamp:.3f}s")
        except Exception as e:
            print(f"[Recorder] Failed to mark event: {e}")
            
    def create_temp_analysis_file(self, time_interval):
        """
        Create a temporary CSV file with the last minute of recorded data for analysis.
        Returns the path to the temporary file, or None if no data is available.
        """
        if not self.recording or not self.csv_file:
            return None
        
        try:
            # Flush current data to disk
            self.csv_file.flush()
            
            # Read the current CSV file
            import pandas as pd
            df = pd.read_csv(self.filename)
            
            if len(df) == 0:
                return None
            
            # Calculate the cutoff time (1 minute ago from the latest timestamp)
            latest_time = df['timestamp'].max()
            cutoff_time = latest_time - time_interval  # 60 seconds = 1 minute
            
            # Filter data from the last minute
            recent_data = df[df['timestamp'] >= cutoff_time]
            
            if len(recent_data) == 0:
                return None
            
            # Create temporary file
            
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(dir=os.path.dirname(self.filename) , mode='w', suffix='.csv', delete=False)
            recent_data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            
            return temp_file.name
            
        except Exception as e:
            print(f"[Recorder] Failed to create temp analysis file: {e}")
            return None

    def close(self):
        self.stop_recording()

