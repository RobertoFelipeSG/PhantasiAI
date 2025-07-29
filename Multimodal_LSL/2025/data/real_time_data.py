import time
import threading
from collections import deque
import serial
from pyfirmata import Arduino, util
from pylsl import StreamInfo, StreamOutlet

from config.config_manager import load_config  # import config loader

class RealTimeData:
    """
    Manages live data acquisition from an Arduino using PyFirmata.
    Streams data over Lab Streaming Layer (LSL).
    """

    # Default analog input pins A0–A5 in PyFirmata format
    DEFAULT_PINS = ['a:0:i', 'a:1:i', 'a:2:i', 'a:3:i', 'a:4:i', 'a:5:i']

    def __init__(self, port=None, channels=None,
                 sample_rate=220, buffer_seconds=2):
        """
        Initialize the live mode acquisition system.

        Parameters:
        - port: Serial port of the Arduino (e.g., "COM3" or "/dev/ttyACM0").
                If None, loaded from config file.
        - channels: List of analog pins to use (PyFirmata format).
        - sample_rate: Number of samples per second.
        - buffer_seconds: Number of seconds to keep in memory.
        """
        # If no port provided, fetch from config
        if port is None:
            cfg = load_config()
            port = cfg.get("arduino_port")

        self.port = port
        self.channels = channels or self.DEFAULT_PINS
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_len = sample_rate * buffer_seconds  # Total samples in buffer

        self.board = None
        self.pins = {}  # Maps channel names to PyFirmata pin objects
        self._iterator = None  # PyFirmata background iterator

        # Buffers to store recent values and timestamps for each channel
        self._buffers = {ch: deque(maxlen=self.buffer_len) for ch in self.channels}
        self._timestamps = {ch: deque(maxlen=self.buffer_len) for ch in self.channels}

        self._outlets = {}  # LSL StreamOutlets for each channel
        self._stop_event = threading.Event()  # Flag for stopping threads
        self._threads = []  # Background threads for acquisition

    def connect(self):
        """
        Connect to the Arduino board and initialize all configured pins and LSL streams.

        Returns:
        - True if connection is successful.
        - False if the serial port cannot be opened.
        """
        try:
            self.board = Arduino(self.port)
            self._iterator = util.Iterator(self.board)
            self._iterator.start()

            for ch in self.channels:
                # Get pin object for analog input
                self.pins[ch] = self.board.get_pin(ch)

                # Create a corresponding LSL stream for each pin
                info = StreamInfo(
                    name=f"EMG_{ch}",
                    type="EMG",
                    channel_count=1,
                    nominal_srate=self.sample_rate,
                    channel_format='float32',
                    source_id=f"uid_{ch}"
                )
                self._outlets[ch] = StreamOutlet(info)

            return True
        except serial.SerialException:
            return False

    @staticmethod
    def is_sensor_connected(pin, threshold=50, samples=10):
        """
        Heuristically determine whether a sensor is connected by reading multiple values.

        Parameters:
        - pin: PyFirmata pin object.
        - threshold: Minimum value considered as "connected".
        - samples: Number of readings to attempt.

        Returns:
        - True if the max observed value exceeds threshold, else False.
        """
        vals = []
        for _ in range(samples):
            v = pin.read()
            if v is not None:
                vals.append(v * 1023)  # Scale to 0–1023 range
        return bool(vals) and max(vals) > threshold

    def _acquire(self, ch):
        """
        Background thread function to acquire data from a single analog channel.

        Parameters:
        - ch: Channel name (e.g., 'a:0:i')
        """
        pin = self.pins[ch]
        outlet = self._outlets[ch]
        index = 0
        interval = 1.0 / self.sample_rate  # Target interval between samples

        while not self._stop_event.is_set():
            start = time.perf_counter()

            raw = pin.read()
            val = int(raw * 1023) if raw is not None else 0  # Convert to integer
            ts = index / self.sample_rate  # Compute timestamp

            # Push value to LSL
            outlet.push_sample([val])

            # Store in internal buffer
            self._buffers[ch].append(val)
            self._timestamps[ch].append(ts)

            index += 1

            # Maintain consistent sampling interval
            elapsed = time.perf_counter() - start
            time.sleep(max(interval - elapsed, 0))

    def start(self):
        """
        Start data acquisition threads for each active channel.
        """
        if not self.board:
            raise RuntimeError("Must call connect() before start()")

        self._stop_event.clear()

        for ch, pin in self.pins.items():
            # Skip channels that don't appear to have sensors connected
            if not self.is_sensor_connected(pin, threshold=10):
                print(f"Warning: Skipping {ch} no signal detected.")
                continue

            # Start a new acquisition thread for this channel
            thread = threading.Thread(target=self._acquire, args=(ch,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self):
        """
        Stop all acquisition threads cleanly.
        """
        self._stop_event.set()
        for thread in self._threads:
            thread.join()
        self._threads.clear()

    def read_latest(self, selected_indices=None):
        """
        Return the most recent value from each channel buffer.

        Parameters:
        - selected_indices: Optional list of indices to filter returned values.

        Returns:
        - List of latest values from all or selected channels.
        """
        values = []
        for ch in self.channels:
            buf = self._buffers[ch]
            values.append(buf[-1] if buf else 0.0)  # Get latest or default to 0

        if selected_indices is not None:
            return [values[i] for i in selected_indices]

        return values

    def close(self):
        """
        Gracefully stop acquisition and disconnect from the Arduino board.
        """
        self.stop()
        if self.board:
            self.board.exit()
            self.board = None
