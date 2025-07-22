from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from data_thread import DataThread

ZOOM_FACTOR = 1.2

class GraphWidget(QtWidgets.QWidget):

    paused       = QtCore.pyqtSignal()
    resumed      = QtCore.pyqtSignal()
    zoomed_in    = QtCore.pyqtSignal()
    zoomed_out   = QtCore.pyqtSignal()

    def __init__(self, read_func, num_channels=1, channel_labels=None,
                 title="EMG Visualization", sample_rate=220,
                 buffer_seconds=2, parent=None):
        super().__init__(parent)
        self._manual_y = False
        self.num_channels = num_channels

        self.last_spike_time = 0
        self.spike_interval = 5.0  # seconds between spike markers
        self.spike_lines = []

        # Channel labels
        if channel_labels and len(channel_labels) == num_channels:
            self.channel_labels = channel_labels
        else:
            self.channel_labels = [f"Ch {i+1}" for i in range(num_channels)]

        # Estimate amplitude range and spacing
        first = np.asarray(read_func())
        amp_range = float(np.max(first) - np.min(first))
        self.spacing = amp_range * 1.2 if amp_range > 0 else 1.0

        # Data acquisition thread
        self.thread = DataThread(read_func, sample_rate, buffer_seconds,
                                 multi_channel=(num_channels > 1))
        self.thread.dataUpdated.connect(self.update_plot)

        # Plot widget setup
        self.plot = pg.PlotWidget()
        self.plot.showAxis('left')
        self.plot.showAxis('bottom')
        self.plot.setBackground("w")
        window = buffer_seconds
        self.plot.setXRange(0, window, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend(offset=(10, 10))

        # Curves and vertical offsets
        self.curves = []
        self.offsets = []
        for idx in range(self.num_channels):
            color = pg.intColor(idx, self.num_channels)
            pen = pg.mkPen(color, width=1.5)
            curve = self.plot.plot(pen=pen, name=self.channel_labels[idx])
            self.curves.append(curve)
            self.offsets.append(idx * self.spacing)

        # Control buttons
        btn_size = QtCore.QSize(160, 60)
        font = QtGui.QFont("Segoe UI", 8)

        self.btn_pause      = QtWidgets.QPushButton("Pause")
        self.btn_zoom_in    = QtWidgets.QPushButton("Zoom In")
        self.btn_zoom_out   = QtWidgets.QPushButton("Zoom Out")
        self.btn_reset_zoom = QtWidgets.QPushButton("Reset Zoom")

        for btn in (self.btn_pause, self.btn_zoom_in,
                    self.btn_zoom_out, self.btn_reset_zoom):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            btn.setStyleSheet("QPushButton { font-size: 9pt; }")

        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_zoom_in)
        control_layout.addWidget(self.btn_zoom_out)
        control_layout.addWidget(self.btn_reset_zoom)
        control_layout.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.plot, stretch=1)
        layout.addLayout(control_layout)

        # Start the data acquisition thread
        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, t, y):
        if t.size == 0 or y.size == 0:
            return

        # Update each curve with its vertical offset
        for idx, curve in enumerate(self.curves):
            curve.setData(t, y[:, idx] + self.offsets[idx])

        # Keep X-axis focused on most recent data
        t_last = t[-1]
        window = self.thread.buffer_len / self.thread.sample_rate
        self.plot.setXRange(t_last - window, t_last, padding=0)

        # Add vertical spike line every 5 seconds
        current_time = t_last
        if current_time - self.last_spike_time >= self.spike_interval:
            # Remove old spike lines outside current window
            for line in self.spike_lines[:]:
                if line.value() < current_time - window:
                    self.plot.removeItem(line)
                    self.spike_lines.remove(line)

            spike_line = pg.InfiniteLine(pos=current_time, angle=90, pen=pg.mkPen('yellow', width=2))
            self.plot.addItem(spike_line)
            self.spike_lines.append(spike_line)
            self.last_spike_time = current_time

        # Auto-range Y if not zoomed manually
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
        vb = self.plot.getViewBox()
        vb.scaleBy((1, 1 / ZOOM_FACTOR))
        self._manual_y = True
        self.zoomed_in.emit()

    def zoom_out(self):
        vb = self.plot.getViewBox()
        vb.scaleBy((1, ZOOM_FACTOR))
        self._manual_y = True
        self.zoomed_out.emit()

    def reset_zoom(self):
        vb = self.plot.getViewBox()
        window = self.thread.buffer_len / self.thread.sample_rate
        vb.setXRange(0, window, padding=0)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._manual_y = False

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.stop()
        super().closeEvent(event)
