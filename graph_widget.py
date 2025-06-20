from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from data_thread import DataThread

ZOOM_FACTOR = 1.2

class GraphWidget(QtWidgets.QWidget):
    misEnPause = QtCore.pyqtSignal()
    repris    = QtCore.pyqtSignal()
    zoomAvant = QtCore.pyqtSignal()
    zoomArriere = QtCore.pyqtSignal()

    def __init__(self, read_func, title="EMG"):
        super().__init__()
        self.thread = DataThread(read_func)

        # ─── Trace ───────────────────────────────────────────────────────────────
        self.plot = pg.PlotWidget(title=title)
        self.plot.setBackground("w")
        self.plot.setXRange(0, 2, padding=0)
        self.plot.setStyleSheet("border-radius: 8px; border: 1px solid #cccccc;")
        self.plot.setLabel('bottom', 'Temps (en secondes)', **{'size':'10pt'})
        self.plot.setLabel('left',   'Amplitude', **{'size':'10pt'})
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.thread.curve = self.plot.plot(pen=pg.mkPen("#A523DC", width=2))

        # ─── Boutons ─────────────────────────────────────────────────────────────
        self.btn_pause    = QtWidgets.QPushButton("Pause")
        self.btn_zoom_in  = QtWidgets.QPushButton("+")
        self.btn_zoom_out = QtWidgets.QPushButton("–")
        self.btn_reset_zoom = QtWidgets.QPushButton("Reset")  
        for btn in (self.btn_pause, self.btn_zoom_in, self.btn_zoom_out, self.btn_reset_zoom):  # Update this line
            btn.setFixedHeight(28)
            btn.setFont(QtGui.QFont("", 9))
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)  

        ctl_layout = QtWidgets.QHBoxLayout()
        ctl_layout.addWidget(self.btn_pause)
        ctl_layout.addWidget(self.btn_zoom_in)
        ctl_layout.addWidget(self.btn_zoom_out)
        ctl_layout.addWidget(self.btn_reset_zoom)  
        ctl_layout.addStretch()
        ctl_layout.setSpacing(6)
        ctl_layout.setContentsMargins(0,0,0,0)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4,4,4,4)
        v.addWidget(self.plot, stretch=1)
        v.addLayout(ctl_layout)

        # ─── Démarrage du thread ─────────────────────────────────────────────────
        self.thread.dataUpdated.connect(self.thread.update_plot)
        self.thread.start()
        self._manual_y = False

    def toggle_pause(self):
        if self.thread.running:
            self.thread.stop()
            self.btn_pause.setText("Reprendre")
            self.misEnPause.emit()
        else:
            if not self.thread.running: 
                self.thread.start()
            self.btn_pause.setText("Pause")
            self.repris.emit()

    def zoom_in(self):
        vb = self.plot.getViewBox()
        vb.scaleBy((1, 1/ZOOM_FACTOR))
        self.thread._manual_y = True
        self.zoomAvant.emit()

    def zoom_out(self):
        vb = self.plot.getViewBox()
        vb.scaleBy((1, ZOOM_FACTOR))
        self.thread._manual_y = True
        self.zoomArriere.emit()

    def reset_zoom(self):
        vb = self.plot.getViewBox()
        vb.setXRange(0, 2, padding=0)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.thread._manual_y = False

    def closeEvent(self, evt):
        if self.thread.running:
            self.thread.stop()
        super().closeEvent(evt)
