import time
import csv
import numpy as np
import os

class EMGRecorder:
    def __init__(self, parent):
        self.parent = parent
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.event_times = []
        self.filename = None
        self._index = 0
        self.channel_count = 0  # New

    def start_recording(self):
        if self.recording:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(os.path.dirname(__file__), f"emg_recording_{timestamp}.csv")

        try:
            self.csv_file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)

            # Get channel count from parent if available
            self.channel_count = getattr(self.parent, 'num_channels', 1)

            header = ['timestamp'] + [f'ch{i+1}' for i in range(self.channel_count)] + ['event']
            self.csv_writer.writerow(header)
            self.csv_file.flush()
            self.recording = True
            self.parent.chat.log_event(f"Recording started: {self.filename}")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False

    def record_data_point(self, timestamp, emg_values):
        if not self.recording or not self.csv_writer:
            return

        try:
            if isinstance(emg_values, (np.ndarray, list)):
                emg_values = [float(v) for v in emg_values]
            else:
                emg_values = [float(emg_values)]

            event_flag = any(abs(timestamp - t) < 0.001 for t in self.event_times)

            row = [f"{timestamp:.6f}"] + [f"{v:.6f}" for v in emg_values] + [int(event_flag)]
            self.csv_writer.writerow(row)

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

        if hasattr(self.parent, 'graph') and hasattr(self.parent.graph, 'thread'):
            current_time = self.parent.graph.thread._index / self.parent.graph.thread.sample_rate
        else:
            current_time = time.time()

        self.event_times.append(current_time)
        self.parent.chat.log_event(f"Event marked at {current_time:.3f}s")

    def close(self):
        self.stop_recording()
