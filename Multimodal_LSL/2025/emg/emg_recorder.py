import time
import csv
import numpy as np
import os
from ..peaks.peak import analyze_emg_peaks

class EMGRecorder:
    def __init__(self, parent):
        self.parent = parent
        self.recording = False                # Is recording active
        self.csv_file = None                  # File handle
        self.csv_writer = None                # CSV writer
        self.event_times = []                 # Manually marked timestamps
        self.filename = None                  # Full path to file
        self._index = 0                       # Sample counter
        self.channel_count = 0                # Number of channels

    def start_recording(self):
        """Start a new recording session and write the CSV header."""
        if self.recording:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Go one directory up from current file
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Path to "emg-recordings" folder one level up
        recordings_dir = os.path.join(parent_dir, "emg-recordings")

        # Create it if it doesn't exist
        os.makedirs(recordings_dir, exist_ok=True)

        # Full path to CSV file inside recordings folder
        self.filename = os.path.join(recordings_dir, f"emg_recording_{timestamp}.csv")

        try:
            self.csv_file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)

            self.channel_count = getattr(self.parent.graph, 'num_channels', 1)
            header = ['timestamp'] + [f'ch{i+1} (µV)' for i in range(self.channel_count)] + ['event']
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.recording = True
            self.parent.chat.log_event(f"Recording started: {os.path.basename(self.filename)}")
        except Exception as e:
            print(f"[Recorder] Failed to start recording: {e}")
            self.recording = False


    def record_data_point(self, timestamp, emg_values):
        """Record a single EMG sample with optional event flag."""
        if not self.recording or not self.csv_writer:
            return

        try:
            # Convert to µV from V
            if isinstance(emg_values, (np.ndarray, list)):
                emg_values = [float(v) * 1_000_000 for v in emg_values]
            else:
                emg_values = [float(emg_values) * 1_000_000]

            # Check if this timestamp matches a marked event
            event_flag = any(abs(timestamp - t) < 0.001 for t in self.event_times)

            # Build and write row
            emg_values = emg_values[:self.channel_count]
            row = [f"{timestamp:.8f}"] + [f"{v:.5f}" for v in emg_values] + [int(event_flag)]
            self.csv_writer.writerow(row)

            # Periodic flush
            if self._index % 10 == 0:
                self.csv_file.flush()
            self._index += 1
        except Exception as e:
            print(f"[Recorder] Failed to record data point: {e}")

    def stop_recording(self):
        """Safely stop recording and close file."""
        if not self.recording:
            return
        try:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.csv_writer = None
            self.recording = False
            self.parent.chat.log_event("Recording stopped")
            analyze_emg_peaks()

        except Exception as e:
            print(f"[Recorder] Failed to stop recording: {e}")

    def mark_event(self):
        """Mark the current timestamp for event flagging."""
        if not self.recording:
            return
        try:
            if hasattr(self.parent, 'graph') and hasattr(self.parent.graph, 'thread'):
                current_time = self.parent.graph.thread._index / self.parent.graph.thread.sample_rate
            else:
                current_time = time.time()

            self.event_times.append(current_time)
            self.parent.chat.log_event(f"Event marked at {current_time:.3f}s")
        except Exception as e:
            print(f"[Recorder] Failed to mark event: {e}")

    def close(self):
        """Ensure recording is stopped before exiting."""
        self.stop_recording()
