from PyQt5 import QtCore
import numpy as np
import time

class DataThread(QtCore.QThread):
    """
    QThread-based real-time data acquisition thread supporting multi-channel streams.
    Emits:
      dataUpdated(times: np.ndarray, values: np.ndarray)
    """

    # Signal emitted when new data is available.
    # It passes two numpy arrays: timestamps and corresponding values.
    dataUpdated = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    
    # Signal emitted when a marker should be added (synchronized with data timing)
    markerRequested = QtCore.pyqtSignal(float)
    
    # Signal emitted when auto analysis should be performed (synchronized with data timing)
    analysisRequested = QtCore.pyqtSignal(float)  # timestamp

    def __init__(self, read_func, sample_rate=220, buffer_seconds=2, multi_channel=False):
        """
        Initialize the data thread.

        Parameters:
        - read_func: A function to read a single data sample (can return scalar or array).
        - sample_rate: Number of samples per second.
        - buffer_seconds: Length of the data buffer in seconds.
        - multi_channel: Whether the incoming data is multi-channel.
        """
        super().__init__()
        self.read = read_func
        self.sample_rate = sample_rate
        self.buffer_len = int(sample_rate * buffer_seconds)  # Total number of samples to store
        self.multi_channel = multi_channel

        # Get the first sample to determine how many channels are present
        first = self.read()
        arr = np.asarray(first)

        if self.multi_channel:
            self.num_channels = arr.size
            self._values = np.zeros((self.buffer_len, self.num_channels), dtype=float)
        else:
            self.num_channels = 1
            self._values = np.zeros(self.buffer_len, dtype=float)

        # Initialize timestamp array
        self._times = np.zeros(self.buffer_len, dtype=float)

        # Internal index to track current write position
        self._index = 0

        # Control flag for running the thread
        self._running = False
        
        # Marker timing system (synchronized with data) TO BE CHANGED!!!
        self.marker_interval = 5.0  # seconds between markers
        self.last_marker_time = 0.0  # time of last marker
        self.auto_markers_enabled = False
        
        # Auto analysis timing system (synchronized with data) TO BE CHANGED!!!
        self.analysis_interval = 60.0  # seconds between auto analysis
        self.last_analysis_time = 0.0  # time of last analysis
        self.auto_analysis_enabled = False

        # Store the first sample into the buffer
        self._store_sample(arr)

    def _store_sample(self, arr):
        """
        Store a single sample and its corresponding timestamp into the circular buffer.
        """
        pos = self._index % self.buffer_len  # Circular buffer position
        if self.multi_channel:
            self._values[pos, :] = arr.flatten()  # Store all channels
        else:
            self._values[pos] = float(arr)  # Store single value

        # Calculate and store timestamp for the sample
        current_time = self._index / self.sample_rate
        self._times[pos] = current_time
        
        # Check if it's time for a marker (synchronized with data timing) CAMBIAR!
        if (self.auto_markers_enabled and 
            current_time - self.last_marker_time >= self.marker_interval):
            self.last_marker_time = current_time
            self.markerRequested.emit(current_time)
        
        # Check if it's time for auto analysis (synchronized with data timing)
        if (self.auto_analysis_enabled and 
            current_time - self.last_analysis_time >= self.analysis_interval):
            self.last_analysis_time = current_time
            self.analysisRequested.emit(current_time)
        
        self._index += 1  # Increment sample index

    def run(self):
        """
        Main thread loop that reads data, stores it, and emits the buffer contents.
        This runs in a separate thread to allow real-time data acquisition.
        """
        self._running = True
        t_prev = time.perf_counter()  # Timestamp of the previous loop iteration

        while self._running:
            # Read new sample and store it
            arr = np.asarray(self.read())
            self._store_sample(arr)

            # Prepare the output window of data
            if self._index < self.buffer_len:
                # If buffer is not full yet, just use the available portion
                times = self._times[:self._index].copy()
                vals = self._values[:self._index].copy()
            else:
                # If buffer is full, roll to maintain continuous window
                pos = (self._index - 1) % self.buffer_len
                shift = pos + 1
                times = np.roll(self._times, -shift)
                vals = np.roll(self._values, -shift, axis=0)

            # Ensure values are in 2D shape for consistency
            if not self.multi_channel:
                vals = vals.reshape(-1, 1)

            # Emit the new data to any connected slots
            self.dataUpdated.emit(times, vals)

            # Maintain sampling rate by adjusting sleep duration
            dt = time.perf_counter() - t_prev
            time.sleep(max(0, 1/self.sample_rate - dt))
            t_prev = time.perf_counter()

    def stop(self):
        """
        Stop the thread gracefully.
        """
        self._running = False
        self.wait()  # Wait for the thread to finish
    
    def set_marker_interval(self, interval):
        """Set the interval between automatic markers (in seconds)."""
        self.marker_interval = float(interval)
    
    def set_auto_markers_enabled(self, enabled):
        """Enable or disable automatic markers."""
        self.auto_markers_enabled = enabled
        if enabled:
            # Reset marker timing to start fresh
            self.last_marker_time = self._index / self.sample_rate
    
    def get_current_time(self):
        """Get the current synchronized time."""
        return self._index / self.sample_rate
    
    def set_analysis_interval(self, interval):
        """Set the interval between automatic analysis (in seconds)."""
        self.analysis_interval = float(interval)
    
    def set_auto_analysis_enabled(self, enabled):
        """Enable or disable automatic analysis."""
        self.auto_analysis_enabled = enabled
        if enabled:
            # Reset analysis timing to start fresh
            self.last_analysis_time = self._index / self.sample_rate
