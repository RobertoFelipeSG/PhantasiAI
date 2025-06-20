from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import time

from data_thread import DataThread  # separate file: data_thread.py

ZOOM_FACTOR = 1.2

class GraphWidget(QtWidgets.QWidget):

    misEnPause   = QtCore.pyqtSignal()
    repris       = QtCore.pyqtSignal()
    zoomAvant    = QtCore.pyqtSignal()
    zoomArriere  = QtCore.pyqtSignal()

    def __init__(self, read_func, num_channels=1, channel_labels=None,
                 title="EMG Temps Réel", sample_rate=220,
                 buffer_seconds=2, parent=None):
       
        super().__init__(parent)
        self._manual_y = False
        self.num_channels = num_channels
        
        if channel_labels and len(channel_labels) == num_channels:
            self.channel_labels = channel_labels
        else:
            self.channel_labels = [f"Ch {i+1}" for i in range(num_channels)]

        # Data thread
        self.thread = DataThread(read_func, sample_rate, buffer_seconds,
                                 multi_channel=(num_channels > 1))
        self.thread.dataUpdated.connect(self.update_plot)

        # Plot setup
        self.plot = pg.PlotWidget(title=title)
        self.plot.setBackground("w")
        window = buffer_seconds
        self.plot.setXRange(0, window, padding=0)
        self.plot.setStyleSheet("border-radius: 8px; border: 1px solid #cccccc;")
        self.plot.setLabel('bottom', 'Temps', **{'size':'10pt'})
        self.plot.setLabel('left',   'Amplitude', **{'size':'10pt'})
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Legend
        self.legend = self.plot.addLegend()
        columns = min(self.num_channels, 8)
        self.legend.setColumnCount(columns)

        # Create curves with labels
        self.curves = []
        for idx, label in enumerate(self.channel_labels):
            pen = pg.mkPen(pg.intColor(idx, self.num_channels), width=2)
            curve = self.plot.plot(pen=pen, name=label)
            self.curves.append(curve)

        btn_size = QtCore.QSize(120, 60)  # Increased button size
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
    

        self.btn_pause      = QtWidgets.QPushButton("Pause")
        self.btn_zoom_in    = QtWidgets.QPushButton("+")
        self.btn_zoom_out   = QtWidgets.QPushButton("–")
        self.btn_reset_zoom = QtWidgets.QPushButton("Reset")
        
        for btn in (self.btn_pause, self.btn_zoom_in,
                    self.btn_zoom_out, self.btn_reset_zoom):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            # Add style sheet for additional control
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 8pt;
                    padding: 8px;
                }
            """)
        
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)

        # Layout
        ctl_layout = QtWidgets.QHBoxLayout()
        ctl_layout.setContentsMargins(0,0,0,0)
        ctl_layout.setSpacing(8)
        ctl_layout.addWidget(self.btn_pause)
        ctl_layout.addWidget(self.btn_zoom_in)
        ctl_layout.addWidget(self.btn_zoom_out)
        ctl_layout.addWidget(self.btn_reset_zoom)
        ctl_layout.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)
        layout.addWidget(self.plot, stretch=1)
        layout.addLayout(ctl_layout)

        # Start acquisition
        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, t, y):
        if t.size == 0 or y.size == 0:
            return
        for idx, curve in enumerate(self.curves):
            curve.setData(t, y[:, idx])
        t_last = t[-1]
        vb = self.plot.getViewBox()
        window = self.thread.buffer_len / self.thread.sample_rate
        xmin = max(0, t_last - window)
        vb.setXRange(xmin, xmin + window, padding=0)
        if not self._manual_y:
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def toggle_pause(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_pause.setText("Reprendre")
            self.misEnPause.emit()
        else:
            self.thread.start()
            self.btn_pause.setText("Pause")
            self.repris.emit()

    def zoom_in(self):
        vb = self.plot.getViewBox()
        vb.scaleBy((1, 1/ZOOM_FACTOR))
        self._manual_y = True
        self.zoomAvant.emit()

    def zoom_out(self):
        vb = self.plot.getViewBox()
        vb.scaleBy((1, ZOOM_FACTOR))
        self._manual_y = True
        self.zoomArriere.emit()

    def reset_zoom(self):
        vb = self.plot.getViewBox()
        window = self.thread.buffer_len / self.thread.sample_rate
        vb.setXRange(0, window, padding=0)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._manual_y = False

    def closeEvent(self, evt):
        if self.thread.isRunning():
            self.thread.stop()
        super().closeEvent(evt)
