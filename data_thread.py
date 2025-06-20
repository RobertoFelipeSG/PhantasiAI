import threading
import time
from collections import deque

import numpy as np
from PyQt5 import QtCore
import pyqtgraph as pg

SAMPLE_RATE    = 220       
BUFFER_SECONDS = 2         

class DataThread(QtCore.QObject):
    dataUpdated = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, read_func):
        super().__init__()
        self.read = read_func
        maxlen = BUFFER_SECONDS * SAMPLE_RATE
        self.buf, self.tbuf = deque(maxlen=maxlen), deque(maxlen=maxlen)
        self.idx, self.running = 0, False

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            t0 = time.perf_counter()
            ts = self.idx / SAMPLE_RATE
            val = self.read()
            self.buf.append(val)
            self.tbuf.append(ts)
            self.idx += 1

            self.dataUpdated.emit(
                np.array(self.tbuf, dtype=float),
                np.array(self.buf, dtype=float),
            )

            dt = time.perf_counter() - t0
            time.sleep(max(1/SAMPLE_RATE - dt, 0))

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, t, y):
        if t.size == 0:
            return
        self.curve.setData(t, y)
        t_last = t[-1]
        vb = self.curve.getViewBox()
        xmin = max(0, t_last - BUFFER_SECONDS)
        vb.setXRange(xmin, xmin + BUFFER_SECONDS, padding=0)
        if not getattr(self, '_manual_y', False):
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)
