import time
import threading
from collections import deque
import serial
from pyfirmata import Arduino, util
from pylsl import StreamInfo, StreamOutlet

class LiveMode:
    """
    Manages live data acquisition from an Arduino using PyFirmata.
    Streams data over Lab Streaming Layer (LSL).
    """

    DEFAULT_PINS = ['a:0:i', 'a:1:i', 'a:2:i', 'a:3:i', 'a:4:i', 'a:5:i']

    def __init__(self, port="COM3", channels=None,
                 sample_rate=220, buffer_seconds=2):
        self.port = port
        self.channels = channels or self.DEFAULT_PINS
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_len = sample_rate * buffer_seconds

        self.board = None
        self.pins = {}
        self._iterator = None

        self._buffers = {ch: deque(maxlen=self.buffer_len) for ch in self.channels}
        self._timestamps = {ch: deque(maxlen=self.buffer_len) for ch in self.channels}

        self._outlets = {}
        self._stop_event = threading.Event()
        self._threads = []

    def connect(self):
        """Connect to the Arduino and initialize LSL streams."""
        try:
            self.board = Arduino(self.port)
            self._iterator = util.Iterator(self.board)
            self._iterator.start()

            for ch in self.channels:
                self.pins[ch] = self.board.get_pin(ch)
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
        """Check if a sensor is connected and producing data above a threshold."""
        vals = []
        for _ in range(samples):
            v = pin.read()
            if v is not None:
                vals.append(v * 1023)
        return bool(vals) and max(vals) > threshold

    def _acquire(self, ch):
        """Data acquisition loop for a single channel."""
        pin = self.pins[ch]
        outlet = self._outlets[ch]
        index = 0
        interval = 1.0 / self.sample_rate

        while not self._stop_event.is_set():
            start = time.perf_counter()

            raw = pin.read()
            val = int(raw * 1023) if raw is not None else 0
            ts = index / self.sample_rate

            outlet.push_sample([val])
            self._buffers[ch].append(val)
            self._timestamps[ch].append(ts)

            index += 1
            elapsed = time.perf_counter() - start
            time.sleep(max(interval - elapsed, 0))

    def start(self):
        """Start acquiring data on all connected channels."""
        if not self.board:
            raise RuntimeError("Must call connect() before start()")

        self._stop_event.clear()
        for ch, pin in self.pins.items():
            if not self.is_sensor_connected(pin, threshold=10):
                print(f"Warning: Skipping {ch} — no signal detected.")
                continue

            thread = threading.Thread(target=self._acquire, args=(ch,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self):
        """Stop data acquisition."""
        self._stop_event.set()
        for thread in self._threads:
            thread.join()
        self._threads.clear()

    def read_latest(self, selected_indices=None):
        """Read the most recent value from each channel."""
        values = []
        for ch in self.channels:
            buf = self._buffers[ch]
            values.append(buf[-1] if buf else 0.0)

        if selected_indices is not None:
            return [values[i] for i in selected_indices]
        return values

    def close(self):
        """Stop all activity and safely disconnect the board."""
        self.stop()
        if self.board:
            self.board.exit()
            self.board = None
