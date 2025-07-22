from PyQt5 import QtCore
import numpy as np
import time

class DataThread(QtCore.QThread):
    """
    QThread-based real-time data acquisition thread supporting multi-channel streams.
    Emits:
      dataUpdated(times: np.ndarray, values: np.ndarray)
    """
    dataUpdated = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, read_func, sample_rate=220, buffer_seconds=2, multi_channel=False):
        super().__init__()
        self.read = read_func
        self.sample_rate = sample_rate
        self.buffer_len = int(sample_rate * buffer_seconds)
        self.multi_channel = multi_channel

        # Infer channel count from first read
        first = self.read()
        arr = np.asarray(first)
        if self.multi_channel:
            self.num_channels = arr.size
            self._values = np.zeros((self.buffer_len, self.num_channels), dtype=float)
        else:
            self.num_channels = 1
            self._values = np.zeros(self.buffer_len, dtype=float)
        self._times = np.zeros(self.buffer_len, dtype=float)

        # Initialize buffer
        self._index = 0
        self._running = False
        self._store_sample(arr)

    def _store_sample(self, arr):
        pos = self._index % self.buffer_len
        if self.multi_channel:
            self._values[pos, :] = arr.flatten()
        else:
            self._values[pos] = float(arr)
        self._times[pos] = self._index / self.sample_rate
        self._index += 1

    def run(self):
        self._running = True
        t_prev = time.perf_counter()
        while self._running:
            arr = np.asarray(self.read())
            self._store_sample(arr)

            # Build window
            if self._index < self.buffer_len:
                times = self._times[:self._index].copy()
                vals = self._values[:self._index].copy()
            else:
                pos = (self._index - 1) % self.buffer_len
                shift = pos + 1
                times = np.roll(self._times, -shift)
                vals = np.roll(self._values, -shift, axis=0)

            # Ensure 2D array
            if not self.multi_channel:
                vals = vals.reshape(-1, 1)

            # Emit
            self.dataUpdated.emit(times, vals)

            # Maintain rate
            dt = time.perf_counter() - t_prev
            time.sleep(max(0, 1/self.sample_rate - dt))
            t_prev = time.perf_counter()

    def stop(self):
        self._running = False
        self.wait()
