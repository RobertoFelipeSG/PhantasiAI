import time
import csv
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from sibling directories
from utils.path_utils import add_parent_to_syspath
add_parent_to_syspath(1) 

from emg.emg_peak_analyzer import EMGPeakAnalyzer

class EMGRecorder:
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

    def start_recording(self):
        if self.recording:
            return

        timestamp = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
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
            self.parent.chat.log_event(f"Recording started: {os.path.basename(self.filename)}")
        except Exception as e:
            print(f"[Recorder] Failed to start recording: {e}")
            self.recording = False

    def record_data_point(self, timestamp, emg_values):
        if not self.recording or not self.csv_writer:
            return

        try:
            if isinstance(emg_values, (np.ndarray, list)):
                emg_values = [float(v) * 1_000_000 for v in emg_values]
            else:
                emg_values = [float(emg_values) * 1_000_000]

            event_flag = any(abs(timestamp - t) < 0.001 for t in self.event_times)
            emg_values = emg_values[:self.channel_count]
            row = [f"{timestamp:.8f}"] + [f"{v:.5f}" for v in emg_values] + [int(event_flag)]
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

            if self.filename:
                analyzer = EMGPeakAnalyzer(csv_path=self.filename)
                analyzer.run(show_plots=False, save_results=True)

        except Exception as e:
            print(f"[Recorder] Failed to stop recording: {e}")

    def mark_event(self):
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
        self.stop_recording()
