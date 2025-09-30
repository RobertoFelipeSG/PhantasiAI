from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from data.data_thread import DataThread

# Adjust this factor for how fast the Y-axis zooms
ZOOM_FACTOR = 1.5

class GraphWidget(QtWidgets.QWidget):
    """
    Widget for real-time EMG data visualization using pyqtgraph.
    Supports zooming, pause/resume, and spike markers every 5 seconds.
    """

    # Custom signals for external communication
    paused      = QtCore.pyqtSignal()
    resumed     = QtCore.pyqtSignal()
    zoomed_in   = QtCore.pyqtSignal()
    zoomed_out  = QtCore.pyqtSignal()
    countdown_updated = QtCore.pyqtSignal(str)  # Signal to update countdown in main window
    automatic_marker_added = QtCore.pyqtSignal(float)  # signal when automatic marker is added (timestamp)

    def __init__(self, read_func, num_channels=1, channel_labels=None,
                 title="EMG Visualization", sample_rate=220,
                 buffer_seconds=2, parent=None):
        super().__init__(parent)

        self._manual_y = False           # Flag to disable auto Y-range after manual zoom
        self.num_channels = num_channels
        self.scale_to_µV = True          # Whether to convert from volts to microvolts

        self.last_spike_time = 0         # Time when last spike marker was added
        self.spike_interval = 5.0        # Add spike every N seconds
        self.spike_lines = []            # Store spike lines for cleanup
        self.auto_spike_enabled = True   # Whether automatic spikes are enabled
        
        # Timer system for marker countdown (now synchronized with data thread)
        self.countdown_timer = QtCore.QTimer()
        self.countdown_timer.timeout.connect(self.update_marker_countdown)
        self.marker_countdown = 0        # Current countdown value in seconds
        
        # Store reference to main window to acces the recorder 
        self.main_window = parent

        # Set channel labels or default to "Ch 1", "Ch 2", etc.
        self.channel_labels = channel_labels if (
            channel_labels and len(channel_labels) == num_channels
        ) else [f"Ch {i+1}" for i in range(num_channels)]

        # Estimate amplitude spacing from first data read
        first = np.asarray(read_func())
        amp_range = float(np.max(first) - np.min(first))
        self.spacing = amp_range * 1.2 if amp_range > 0 else 1.0

        # Create data thread to read EMG data asynchronously
        self.thread = DataThread(read_func, sample_rate, buffer_seconds,
                                 multi_channel=(num_channels > 1))
        self.thread.dataUpdated.connect(self.update_plot)
        self.thread.markerRequested.connect(self.add_automatic_marker)

        # Create main plot widget
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel('left', 'Amplitude', units='µV')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setXRange(0, buffer_seconds, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend(offset=(10, 10))

        # Create one curve per channel with color and vertical offset
        self.curves = []
        self.offsets = []
        
        # Custom color sequence: blue, violet, red, yellow, green, cyan, orange, dark blue, etc.
        custom_colors = [
            (0, 0, 255),      # Blue
            (138, 43, 226),   # Violet (BlueViolet)
            (255, 0, 0),      # Red
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Green
            (0, 255, 255),    # Cyan
            (255, 165, 0),    # Orange
            (0, 0, 139),      # Dark Blue
            (139, 0, 0),      # Dark Red
            (0, 139, 0),      # Dark Green
        ]
        
        for idx in range(self.num_channels):
            # Use custom colors, cycling through if more channels than colors
            color_rgb = custom_colors[idx % len(custom_colors)]
            color = pg.mkColor(color_rgb)
            
            pen = pg.mkPen(color, width=1.5)
            curve = self.plot.plot(pen=pen, name=self.channel_labels[idx])
            self.curves.append(curve)
            self.offsets.append(idx * self.spacing)

        # --- Control Buttons ---
        btn_size = QtCore.QSize(70, 30) #160, 60
        font = QtGui.QFont("Segoe UI", 2) #8
        
        self.btn_pause      = QtWidgets.QPushButton("Pause")
        self.btn_zoom_in    = QtWidgets.QPushButton("+")
        self.btn_zoom_out   = QtWidgets.QPushButton("–")
        self.btn_reset_zoom = QtWidgets.QPushButton("Auto +/-")

        # Apply consistent style to all buttons
        for btn in (self.btn_pause, self.btn_zoom_in,
                    self.btn_zoom_out, self.btn_reset_zoom):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            btn.setStyleSheet("QPushButton { font-size: 9pt; }")

        # Connect button actions
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)

        # Layout buttons horizontally
        ctl_layout = QtWidgets.QHBoxLayout()
        ctl_layout.setContentsMargins(0, 0, 0, 0)
        ctl_layout.setSpacing(8)
        ctl_layout.addWidget(self.btn_pause)
        ctl_layout.addWidget(self.btn_zoom_in)
        ctl_layout.addWidget(self.btn_zoom_out)
        ctl_layout.addWidget(self.btn_reset_zoom)
        ctl_layout.addStretch()

        # Overall layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.plot, stretch=1)
        layout.addLayout(ctl_layout)

        # Start data acquisition thread
        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, t, y):
        """
        Update the graph with new time `t` and signal `y` values.
        Applies per-channel vertical offset and auto scrolls the X-axis.
        """
        if t.size == 0 or y.size == 0:
            return

        if self.scale_to_µV:
            y = y * 1_000_000  # Convert V → µV

        # Update each channel's curve
        for idx, curve in enumerate(self.curves):
            curve.setData(t, y[:, idx] + self.offsets[idx])

        # Auto-scroll X-axis to follow newest data
        t_last = t[-1]
        window = self.thread.buffer_len / self.thread.sample_rate
        self.plot.setXRange(t_last - window, t_last, padding=0)
        
        # Clean up old spike lines that are outside the visible window
        for line in self.spike_lines[:]:
            if line.value() < t_last - window:
                self.plot.removeItem(line)
                self.spike_lines.remove(line)

        # Automatically scale Y unless user manually zoomed
        if not self._manual_y:
            vb = self.plot.getViewBox()
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def toggle_pause(self):
        """
        Start or stop the data acquisition thread.
        Updates button text and emits custom signals.
        """
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_pause.setText("Resume")
            self.paused.emit()
        else:
            self.thread.start()
            self.btn_pause.setText("Pause")
            self.resumed.emit()

    def zoom_in(self):
        """
        Zoom in (reduce Y-axis range).
        """
        self._adjust_y_range(scale=1 / ZOOM_FACTOR)
        self._manual_y = True
        self.zoomed_in.emit()

    def zoom_out(self):
        """
        Zoom out (increase Y-axis range).
        """
        self._adjust_y_range(scale=ZOOM_FACTOR)
        self._manual_y = True
        self.zoomed_out.emit()

    def reset_zoom(self):
        """
        Reset zoom to auto-ranging mode.
        """
        vb = self.plot.getViewBox()
        window = self.thread.buffer_len / self.thread.sample_rate
        vb.setXRange(0, window, padding=0)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._manual_y = False

    def _adjust_y_range(self, scale=1.0):
        """
        Internal function to manually scale the Y-axis by a factor.
        """
        vb = self.plot.getViewBox()
        y_min, y_max = vb.viewRange()[1]
        y_center = (y_min + y_max) / 2
        y_half = (y_max - y_min) / 2 * scale
        vb.setYRange(y_center - y_half, y_center + y_half, padding=0)
    
    def add_marker(self, t, label=None):
        """
        Add a vertical line marker at time `t`.
        Optionally display a label in the log.
        """
        spike_line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen('pink', width=2))
        self.plot.addItem(spike_line)
        self.spike_lines.append(spike_line)
        
        # Mark event in recorder when marker is added
        if hasattr(self.main_window, 'recorder') and self.main_window.recorder.recording:
            self.main_window.recorder.mark_event()
    
    def set_auto_spike_enabled(self, enabled):  ### NOT EFFICIENT
        """Enable or disable automatic spike markers."""
        self.auto_spike_enabled = enabled
        if hasattr(self, 'thread'):
            self.thread.set_auto_markers_enabled(enabled)
        
        if enabled:
            # Start countdown timer for display
            self.countdown_timer.start(1000)
            self.marker_countdown = int(self.spike_interval)
            self.update_marker_countdown()
        else:
            # Stop countdown timer
            self.countdown_timer.stop()
            self.countdown_updated.emit("Marker: OFF")
    
    def set_spike_interval(self, interval):
        """Set the interval for automatic spike markers."""
        self.spike_interval = float(interval)
        if hasattr(self, 'thread'):
            self.thread.set_marker_interval(interval)
    
    def update_marker_countdown(self):
        """Update the marker countdown display."""
        if self.auto_spike_enabled and hasattr(self, 'thread') and self.thread.isRunning():
            # Calculate time until next marker using data thread timing
            current_time = self.thread.get_current_time()
            time_since_last = current_time - self.last_spike_time   ### NOT EFFICIENT
            time_until_next = self.spike_interval - time_since_last ### NOT EFFICIENT
            
            if time_until_next > 0:
                countdown_seconds = int(time_until_next) + 1
                self.countdown_updated.emit(f"Marker: {countdown_seconds}s")
            else:
                self.countdown_updated.emit("Marker: 0s")
    
    def add_automatic_marker(self, timestamp):
        """Add an automatic marker and emit signal (called by data thread)."""
        # Add visual marker
        spike_line = pg.InfiniteLine(pos=timestamp, angle=90, pen=pg.mkPen('pink', width=2))
        self.plot.addItem(spike_line)
        self.spike_lines.append(spike_line)
        self.last_spike_time = timestamp
        
        # Emit signal for main window
        print(f"[Graph] Adding automatic marker at {timestamp:.3f}s, emitting signal")
        self.automatic_marker_added.emit(timestamp)


    def closeEvent(self, event):
        """
        Ensures the data thread and timers are stopped before the widget is closed.
        """
        if self.thread.isRunning():
            self.thread.stop()
        self.countdown_timer.stop()
        super().closeEvent(event)
