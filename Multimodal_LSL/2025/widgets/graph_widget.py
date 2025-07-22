from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from data.data_thread import DataThread

# Adjust this for more aggressive zoom in/out
ZOOM_FACTOR = 1.5

class GraphWidget(QtWidgets.QWidget):
    # Custom signals to communicate with parent (e.g. for logs)
    paused   = QtCore.pyqtSignal()
    resumed       = QtCore.pyqtSignal()
    zoomed_in    = QtCore.pyqtSignal()
    zoomed_out  = QtCore.pyqtSignal()

    def __init__(self, read_func, num_channels=1, channel_labels=None,
                 title="EMG Visualization", sample_rate=220,
                 buffer_seconds=2, parent=None):
        super().__init__(parent)
        self._manual_y = False  # Tracks if user manually zoomed Y-axis
        self.num_channels = num_channels
        self.scale_to_µV = True  # Convert from V to µV for display

        self.last_spike_time = 0
        self.spike_interval = 5.0  # Add spike line every 5 seconds
        self.spike_lines = []

        # Fallback labels if none provided
        self.channel_labels = channel_labels if (
            channel_labels and len(channel_labels) == num_channels
        ) else [f"Ch {i+1}" for i in range(num_channels)]

        # Estimate vertical spacing from initial data
        first = np.asarray(read_func())
        amp_range = float(np.max(first) - np.min(first))
        self.spacing = amp_range * 1.2 if amp_range > 0 else 1.0

        # Data acquisition thread (async EMG reader)
        self.thread = DataThread(read_func, sample_rate, buffer_seconds,
                                 multi_channel=(num_channels > 1))
        self.thread.dataUpdated.connect(self.update_plot)

        # Setup the plot widget
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel('left', 'Amplitude', units='µV')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setXRange(0, buffer_seconds, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend(offset=(10, 10))

        # Create a curve for each channel with visual offset
        self.curves = []
        self.offsets = []
        for idx in range(self.num_channels):
            color = pg.intColor(idx, self.num_channels)
            pen = pg.mkPen(color, width=1.5)
            curve = self.plot.plot(pen=pen, name=self.channel_labels[idx])
            self.curves.append(curve)
            self.offsets.append(idx * self.spacing)

        # --- Control Buttons ---
        btn_size = QtCore.QSize(160, 60)
        font = QtGui.QFont("Segoe UI", 8)

        self.btn_pause      = QtWidgets.QPushButton("Pause")
        self.btn_zoom_in    = QtWidgets.QPushButton("+")
        self.btn_zoom_out   = QtWidgets.QPushButton("–")
        self.btn_reset_zoom = QtWidgets.QPushButton("Auto Zoom")

        for btn in (self.btn_pause, self.btn_zoom_in,
                    self.btn_zoom_out, self.btn_reset_zoom):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            btn.setStyleSheet("QPushButton { font-size: 9pt; }")

        # Connect buttons to actions
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)

        # Layout: buttons below graph
        ctl_layout = QtWidgets.QHBoxLayout()
        ctl_layout.setContentsMargins(0, 0, 0, 0)
        ctl_layout.setSpacing(8)
        ctl_layout.addWidget(self.btn_pause)
        ctl_layout.addWidget(self.btn_zoom_in)
        ctl_layout.addWidget(self.btn_zoom_out)
        ctl_layout.addWidget(self.btn_reset_zoom)
        ctl_layout.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.plot, stretch=1)
        layout.addLayout(ctl_layout)

        self.thread.start()  # Begin background reading

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, t, y):
        if t.size == 0 or y.size == 0:
            return

        if self.scale_to_µV:
            y = y * 1_000_000  # Convert V → µV


        # Update each channel's curve with vertical offset
        for idx, curve in enumerate(self.curves):
            curve.setData(t, y[:, idx] + self.offsets[idx])

        # Scroll X-axis to keep latest data in view
        t_last = t[-1]
        window = self.thread.buffer_len / self.thread.sample_rate
        self.plot.setXRange(t_last - window, t_last, padding=0)

        # Add vertical marker every 5 seconds
        if t_last - self.last_spike_time >= self.spike_interval:
            for line in self.spike_lines[:]:
                if line.value() < t_last - window:
                    self.plot.removeItem(line)
                    self.spike_lines.remove(line)

            spike_line = pg.InfiniteLine(pos=t_last, angle=90, pen=pg.mkPen('yellow', width=2))
            self.plot.addItem(spike_line)
            self.spike_lines.append(spike_line)
            self.last_spike_time = t_last

        # Auto-range Y unless user zoomed manually
        if not self._manual_y:
            vb = self.plot.getViewBox()
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def toggle_pause(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_pause.setText("Resume")
            self.paused.emit()
        else:
            self.thread.start()
            self.btn_pause.setText("Pause")
            self.resumed.emit()

    def zoom_in(self):
        self._adjust_y_range(scale=1 / ZOOM_FACTOR)
        self._manual_y = True
        self.zoomed_in.emit()

    def zoom_out(self):
        self._adjust_y_range(scale=ZOOM_FACTOR)
        self._manual_y = True
        self.zoomed_out.emit()

    def reset_zoom(self):
        vb = self.plot.getViewBox()
        window = self.thread.buffer_len / self.thread.sample_rate
        vb.setXRange(0, window, padding=0)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._manual_y = False

    def _adjust_y_range(self, scale=1.0):
        vb = self.plot.getViewBox()
        y_min, y_max = vb.viewRange()[1]
        y_center = (y_min + y_max) / 2
        y_half = (y_max - y_min) / 2 * scale
        vb.setYRange(y_center - y_half, y_center + y_half, padding=0)


    def closeEvent(self, event):
        # Make sure thread exits cleanly
        if self.thread.isRunning():
            self.thread.stop()
        super().closeEvent(event)
