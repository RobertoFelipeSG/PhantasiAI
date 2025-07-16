

import sys
import os
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt


import scipy.signal
_original_welch = scipy.signal.welch
def patched_welch(*args, **kwargs):
    if kwargs.get("window") == "hanning":
        kwargs["window"] = "hann"
    return _original_welch(*args, **kwargs)
scipy.signal.welch = patched_welch

from pysiology.electromyography import (
    getMAV, getRMS, getWL, getZC, getIEMG, getWAMP,
    getVAR, getLOG, getPSD, getMNF, getMDF
)

from graph_widget   import GraphWidget
from chat_widget    import ChatWidget
from startup_dialog import StartupDialog
from live_mode      import LiveMode
from emg_recorder import EMGRecorder

APP_STYLESHEET = """
QMainWindow { background:#FFFFFF; }
QPushButton {
    background:#F7F7F7; border:1px solid #CCC; border-radius:6px;
    padding:6px 12px; font-size:14px;
}
QPushButton:hover   { background:#EEEEEE; }
QPushButton:pressed { background:#DDDDDD; }
QDialog,QWidget     { background:#FFFFFF; }
QLabel,QCheckBox,QRadioButton { font-size:13px; }
QToolBar,QStatusBar {
    background:#F7F7F7; border-top:1px solid #DDD; padding:4px;
}
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode: str, mode_label: str = "", data_type: str = "database"):
        super().__init__()
        self.mode          = mode           # "pro"  | "chat"
        self.data_type     = data_type      # "live" | "database"
        self.selected_channels: list[int] = []

        title = "PhantasiAi" + (f" – {mode_label}" if mode_label else "")
        self.setWindowTitle(title)

        self._build_ui()

        self.recorder = EMGRecorder(self)

        if self.data_type == "live":
            self._init_live()
        else:
            if not self.load_npz_file():
                QtWidgets.QMessageBox.critical(self, "Error", "No .npz file selected. Closing.")
                sys.exit(1)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        self.main_layout = QtWidgets.QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_open     = QtWidgets.QPushButton("Open")
        self.btn_save     = QtWidgets.QPushButton("Save")
        self.btn_channels = QtWidgets.QPushButton("Channels")
        self.btn_features = QtWidgets.QPushButton("EMG Data")
        self.btn_record = QtWidgets.QPushButton("Start Recording")

        for b in (self.btn_open, self.btn_save, self.btn_channels, self.btn_features,self.btn_record):
            b.setFixedSize(QtCore.QSize(160, 60))
            b.setFont(QtGui.QFont("Segoe UI", 7))
            b.setStyleSheet("QPushButton { font-size:7pt; padding:7px; }")

        tool = QtWidgets.QHBoxLayout()
        tool.setContentsMargins(12, 12, 12, 6)
        tool.setSpacing(8)
        for b in (self.btn_open, self.btn_save, self.btn_channels, self.btn_features,self.btn_record):
            tool.addWidget(b)
        tool.addStretch()
        self.main_layout.addLayout(tool)

        self.body_layout = QtWidgets.QHBoxLayout()
        self.body_layout.setContentsMargins(12, 6, 12, 12)
        self.body_layout.setSpacing(10)
        self.main_layout.addLayout(self.body_layout)

        self.chat = ChatWidget(mode=self.mode)
        self.body_layout.addWidget(self.chat)

        self.btn_open.clicked.connect(self.on_file_change)
        self.btn_save.clicked.connect(self.save_logs)
        self.btn_channels.clicked.connect(self.show_channel_dialog)
        self.btn_features.clicked.connect(self.compute_features)
        self.btn_record.clicked.connect(self.toggle_recording)

    def toggle_recording(self):
        print(f"Recording state before toggle: {self.recorder.recording}")
        if self.recorder.recording: # check if recording is currently active
            self.recorder.stop_recording()
            self.btn_record.setText("Start Recording")
            print("Recording stopped")
        else:
            self.recorder.start_recording()
            self.btn_record.setText("Stop Recording")
            print("Recording started")
        print(f"Recording state after toggle: {self.recorder.recording}")

    # captures the Enter press to mark events in the recorded dat
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.recorder.mark_event()  
        super().keyPressEvent(event) 

    def _init_live(self):
        self.live = LiveMode(port="/dev/ttyUSB0", sample_rate=220, buffer_seconds=2)
        if not self.live.connect():
            QtWidgets.QMessageBox.critical(
                self, "Connection Error",
                "Failed to connect to Arduino.\nPlease check the cable/port."
            )
            sys.exit(1)

        self.selected_channels = [0] 
        self.setup_live_graph()
    
    def load_npz_file(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a .npz file", "", "NumPy Archive (*.npz)"
        )
        if not path:
            return False

        try:
            archive = np.load(path)
            key = "emg" if "emg" in archive.files else archive.files[0]
            data = archive[key]
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.full_data       = data
            self.sample_rate     = 220
            self.buffer_seconds  = 2
            self.selected_channels = [0]

            self.chat.set_file(path)
            self.chat.log_event(f"Opened file « {os.path.basename(path)} »")
            self.build_database_view()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return False

    def build_database_view(self):
       
        for i in reversed(range(self.body_layout.count())):
            w = self.body_layout.itemAt(i).widget()
            if isinstance(w, GraphWidget):
                if w.thread.isRunning():
                    w.thread.stop()
                w.setParent(None)
                w.deleteLater()

        mat = self.full_data[:, self.selected_channels]
        self.data_index   = 0
        self.num_channels = mat.shape[1]

        def read_vector():
            v = mat[self.data_index]
            self.data_index = (self.data_index + 1) % len(mat)
            return v.tolist()

        labels = [f"Channel {c+1}" for c in self.selected_channels]
        self.graph = GraphWidget(
            read_func       = read_vector,
            num_channels    = self.num_channels,
            channel_labels  = labels,
            title           = "EMG Channels",
            sample_rate     = self.sample_rate,
            buffer_seconds  = self.buffer_seconds
        )
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Data paused"))
        self.graph.repris.connect  (lambda: self.chat.log_event("Data resumed"))

        
        self.body_layout.addWidget(self.graph, 4)
        self.body_layout.addWidget(self.chat , 1)

        chans = ", ".join(str(c+1) for c in self.selected_channels)
        self.chat.log_event(f"Graph initialized for channels: {chans}")

    def on_file_change(self):
        if self.data_type == "database" and self.load_npz_file():
            self.chat.log_event("Data source changed")

    
    def show_channel_dialog(self):
        if self.data_type == "live":
            total_channels = len(self.live.channels)
        else:
            if not hasattr(self, "full_data"):
                QtWidgets.QMessageBox.warning(self, "No data", "Load a file first.")
                return
            total_channels = self.full_data.shape[1]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Channels")
        lay = QtWidgets.QVBoxLayout(dlg)

       
        select_all = QtWidgets.QCheckBox("Select/Deselect All")
        lay.addWidget(select_all)

        grid = QtWidgets.QGridLayout()
        boxes: list[QtWidgets.QCheckBox] = []
        for idx in range(total_channels):
            cb = QtWidgets.QCheckBox(f"Channel {idx+1}")
            cb.setChecked(idx in self.selected_channels)
            boxes.append(cb)
            grid.addWidget(cb, idx//4, idx%4)
        lay.addLayout(grid)

       
        def toggle_all(state):
            for cb in boxes:
                cb.setChecked(bool(state))
        def update_all():
            select_all.blockSignals(True)
            select_all.setChecked(all(cb.isChecked() for cb in boxes))
            select_all.blockSignals(False)

        select_all.stateChanged.connect(toggle_all)
        for cb in boxes:
            cb.stateChanged.connect(update_all)

       
        btns = QtWidgets.QHBoxLayout()
        ok  = QtWidgets.QPushButton("OK")
        can = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        can.clicked.connect(dlg.reject)
        btns.addStretch(); btns.addWidget(ok); btns.addWidget(can)
        lay.addLayout(btns)

       
        if dlg.exec():
            sel = [i for i, cb in enumerate(boxes) if cb.isChecked()]
            if sel:
                self.selected_channels = sel
                self.chat.log_event("Channels selected: " + ", ".join(str(i+1) for i in sel))
                if self.data_type == "live":
                    self.setup_live_graph()
                else:
                    self.build_database_view()

    def setup_live_graph(self):
       
        for i in reversed(range(self.body_layout.count())):
            w = self.body_layout.itemAt(i).widget()
            if w: w.setParent(None)

        self.live.start()

        labels = [f"Channel {i+1}" for i in self.selected_channels]
        self.graph = GraphWidget(
            read_func      = lambda: self.live.read_latest(self.selected_channels),
            num_channels   = len(self.selected_channels),
            channel_labels = labels,
            title          = "Live EMG",
            sample_rate    = self.live.sample_rate,
            buffer_seconds = self.live.buffer_seconds
        )
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Stream paused"))
        self.graph.repris.connect  (lambda: self.chat.log_event("Stream resumed"))

        self.body_layout.addWidget(self.graph, 4)
        self.body_layout.addWidget(self.chat , 1)
        self.chat.log_event("Live mode started. Streaming from Arduino.")

 
    def save_logs(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Logs", "", "Text File (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.chat.get_logs()))
            QtWidgets.QMessageBox.information(self, "Saved", "Logs saved.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def compute_features(self):
        if self.data_type == "live":
            QtWidgets.QMessageBox.information(self, "Info", "Feature extraction only in database mode.")
            return

        mat = self.full_data[:, self.selected_channels]
        fs  = self.sample_rate
        n_seg = 10
        for idx, ch in enumerate(mat.T):
            seg_len = len(ch)//n_seg
            feats = {k: [] for k in ("MAV","RMS","WL","ZC","IEMG","WAMP","VAR","LOG","MNF","MDF")}
            for j in range(n_seg):
                seg = ch[j*seg_len:(j+1)*seg_len].tolist()
                feats["MAV"].append(getMAV(seg))
                feats["RMS"].append(getRMS(seg))
                feats["WL" ].append(getWL(seg))
                feats["ZC" ].append(getZC(seg,threshold=1e-4))
                feats["IEMG"].append(getIEMG(seg))
                feats["WAMP"].append(getWAMP(seg,threshold=1e-4))
                feats["VAR"].append(getVAR(seg))
                try:
                    val = getLOG(seg); feats["LOG"].append(val if np.isfinite(val) else 0)
                except: feats["LOG"].append(0)
                psd,freq = getPSD(seg,fs)
                feats["MNF"].append(getMNF(psd,freq))
                feats["MDF"].append(getMDF(psd,freq))
            txt = ", ".join(f"{k}:{np.mean(v):.2f}" for k,v in feats.items())
            self.chat.log_event(f"Channel {self.selected_channels[idx]+1} Features — {txt}")

    def closeEvent(self, e: QtGui.QCloseEvent):
        if hasattr(self, "graph") and self.graph.thread.isRunning():
            self.graph.thread.stop()
        if hasattr(self, "live"):
            self.live.close()
        super().closeEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    dlg = StartupDialog()
    if dlg.exec():
        mode, data_type = dlg.get_selections()
        win = MainWindow(mode, f"{mode.capitalize()} Mode", data_type)
        win.resize(1400, 800)
        win.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

