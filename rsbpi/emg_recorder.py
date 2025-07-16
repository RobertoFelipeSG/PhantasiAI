
import time
import csv
import numpy as np
from PySide6 import QtCore

class EMGRecorder:
    def __init__(self, parent):
        self.parent = parent
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.event_times = []
        self.filename = None
        self._index = 0  # Add this counter

    def start_recording(self):
        if self.recording:
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f"emg_recording_{timestamp}.csv"
        
        try:
            # Open file in write mode (not append) to ensure fresh start
            self.csv_file = open(self.filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'emg', 'event'])
            self.csv_file.flush()
            self.recording = True
            self.parent.chat.log_event(f"Recording started: {self.filename}")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False

    def record_data_point(self, timestamp, emg_value):
        if not self.recording or not self.csv_writer:
            return
            
        try:
            # Convert to float if it's an array
            if isinstance(emg_value, (list, np.ndarray)):
                emg_value = float(emg_value[0])
                
            event_flag = any(abs(timestamp - event_ts) < 0.001 for event_ts in self.event_times)
            
            # Write the data point
            self.csv_writer.writerow([
                f"{timestamp:.6f}",
                f"{float(emg_value):.6f}", 
                int(event_flag)
            ])
            
            # Flush every 10 samples
            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1
        except Exception as e:
            print(f"Error recording data point: {e}")



    def stop_recording(self):
        if not self.recording:
            return
            
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        self.recording = False
        self.parent.chat.log_event("Recording stopped")


    def mark_event(self):
        if not self.recording:
            return
            
        # Get timestamp from data thread or system clock
        if hasattr(self.parent, 'graph') and hasattr(self.parent.graph, 'thread'):
            current_time = self.parent.graph.thread._index / self.parent.graph.thread.sample_rate
        else:
            current_time = time.time()
            
        self.event_times.append(current_time)
        self.parent.chat.log_event(f"Event marked at {current_time:.3f}s")

    def close(self):
        self.stop_recording()
