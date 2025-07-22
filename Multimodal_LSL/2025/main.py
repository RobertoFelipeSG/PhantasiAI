import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# Patch scipy welch
import scipy.signal
_original_welch = scipy.signal.welch
def patched_welch(*args, **kwargs):
    if kwargs.get('window') == 'hanning':
        kwargs['window'] = 'hann'
    return _original_welch(*args, **kwargs)
scipy.signal.welch = patched_welch

# EMG features
from pysiology.electromyography import (
    getMAV, getRMS, getWL, getZC, getIEMG, getWAMP,
    getVAR, getLOG, getPSD, getMNF, getMDF
)

from graph_widget import GraphWidget
from chat_widget import ChatWidget
from startup_dialog import StartupDialog
from live_mode import LiveMode
from emg_recorder import EMGRecorder

APP_STYLESHEET = """
QMainWindow {
    background-color: #FFFFFF;
}
QPushButton {
    background-color: #F7F7F7;
    border: 1px solid #CCCCCC;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #EEEEEE;
}
QPushButton:pressed {
    background-color: #DDDDDD;
}
QDialog, QWidget {
    background-color: #FFFFFF;
}
QLabel, QCheckBox, QRadioButton {
    font-size: 13px;
}
QToolBar, QStatusBar {
    background-color: #F7F7F7;
    border-top: 1px solid #DDDDDD;
    padding: 4px;
}
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, mode_label="", data_type="database"):
        super().__init__()
        self.mode = mode
        self.data_type = data_type
        self.selected_channels = []

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
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)

        self.main_layout = QtWidgets.QVBoxLayout(container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.btn_open     = QtWidgets.QPushButton("Open")
        self.btn_save     = QtWidgets.QPushButton("Save")
        self.btn_channels = QtWidgets.QPushButton("Channels")
        self.btn_features = QtWidgets.QPushButton("EMG Data")
        self.btn_record   = QtWidgets.QPushButton("Start Recording")

        for btn in (self.btn_open, self.btn_save, self.btn_channels, self.btn_features, self.btn_record):
            btn.setFixedSize(QtCore.QSize(160, 60))
            btn.setFont(QtGui.QFont("Segoe UI", 7))
            btn.setStyleSheet("QPushButton { font-size: 7pt; padding: 7px; }")

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(12, 12, 12, 6)
        toolbar.setSpacing(8)
        for btn in (self.btn_open, self.btn_save, self.btn_channels, self.btn_features, self.btn_record):
            toolbar.addWidget(btn)
        toolbar.addStretch()
        self.main_layout.addLayout(toolbar)

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

    def _init_live(self):
        self.live = LiveMode(port="COM3", sample_rate=220, buffer_seconds=2)
        if not self.live.connect():
            QtWidgets.QMessageBox.critical(
                self, "Connection Error",
                "Failed to connect to Arduino.\nPlease make sure it is plugged in and try again."
            )
            sys.exit(1)

        self.selected_channels = [0]
        self.setup_live_graph()

    def load_npz_file(self):
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
            self.full_data = data
            self.sample_rate = 220
            self.buffer_seconds = 2
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
        self.data_index = 0
        self.num_channels = mat.shape[1]

        def read_vector():
            v = mat[self.data_index]
            self.data_index = (self.data_index + 1) % len(mat)
            return v.tolist()

        labels = [f"Channel {c+1}" for c in self.selected_channels]
        self.graph = GraphWidget(
            read_func=read_vector,
            num_channels=self.num_channels,
            channel_labels=labels,
            title="EMG Channels",
            sample_rate=self.sample_rate,
            buffer_seconds=self.buffer_seconds
        )

        self.graph.thread.dataUpdated.connect(self.handle_data_update)
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Data paused"))
        self.graph.repris.connect(lambda: self.chat.log_event("Data resumed"))

        if self.mode == "pro":
            self.body_layout.addWidget(self.graph, 4)
            self.body_layout.addWidget(self.chat, 1)
        else:
            self.body_layout.addWidget(self.chat, 1)

        chans = ", ".join(str(c+1) for c in self.selected_channels)
        self.chat.log_event(f"Graph initialized for channels: {chans}")

    def setup_live_graph(self):
        for i in reversed(range(self.body_layout.count())):
            w = self.body_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        self.live.start()
        labels = [f"Channel {i+1}" for i in self.selected_channels]

        self.graph = GraphWidget(
            read_func=lambda: self.live.read_latest(self.selected_channels),
            num_channels=len(self.selected_channels),
            channel_labels=labels,
            title="Live EMG",
            sample_rate=self.live.sample_rate,
            buffer_seconds=self.live.buffer_seconds
        )

        self.graph.thread.dataUpdated.connect(self.handle_data_update)
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Stream paused"))
        self.graph.repris.connect(lambda: self.chat.log_event("Stream resumed"))

        if self.mode == "pro":
            self.body_layout.addWidget(self.graph, 4)
            self.body_layout.addWidget(self.chat, 1)
        else:
            self.body_layout.addWidget(self.chat, 1)

        self.chat.log_event("Live mode started. Streaming from Arduino.")

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

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Channels")
        layout = QtWidgets.QVBoxLayout(dialog)

        select_all = QtWidgets.QCheckBox("Select/Deselect All")
        layout.addWidget(select_all)

        grid = QtWidgets.QGridLayout()
        self.checkboxes = []
        for idx in range(total_channels):
            cb = QtWidgets.QCheckBox(f"Channel {idx+1}")
            cb.setChecked(idx in self.selected_channels)
            self.checkboxes.append(cb)
            grid.addWidget(cb, idx // 4, idx % 4)
        layout.addLayout(grid)

        def toggle_all(state):
            for cb in self.checkboxes:
                cb.setChecked(bool(state))

        def update_all():
            all_checked = all(cb.isChecked() for cb in self.checkboxes)
            select_all.setChecked(all_checked)

        select_all.stateChanged.connect(toggle_all)
        for cb in self.checkboxes:
            cb.stateChanged.connect(update_all)

        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            sel = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
            if sel:
                self.selected_channels = sel
                chans = ', '.join(str(i + 1) for i in sel)
                self.chat.log_event(f"Channels selected: {chans}")
                if self.data_type == "live":
                    self.setup_live_graph()
                else:
                    self.build_database_view()

    def compute_features(self):
        if self.data_type == "live":
            QtWidgets.QMessageBox.information(self, "Info", "Feature extraction only in database mode.")
            return

        mat = self.full_data[:, self.selected_channels]
        fs = self.sample_rate
        n_segs = 10
        for idx, ch_data in enumerate(mat.T):
            seg_len = len(ch_data) // n_segs
            feats = {k: [] for k in ("MAV","RMS","WL","ZC","IEMG","WAMP","VAR","LOG","MNF","MDF")}
            for j in range(n_segs):
                seg = ch_data[j*seg_len:(j+1)*seg_len].tolist()
                feats["MAV"].append(getMAV(seg))
                feats["RMS"].append(getRMS(seg))
                feats["WL" ].append(getWL(seg))
                feats["ZC" ].append(getZC(seg, threshold=1e-4))
                feats["IEMG"].append(getIEMG(seg))
                feats["WAMP"].append(getWAMP(seg, threshold=1e-4))
                feats["VAR"].append(getVAR(seg))
                try:
                    val = getLOG(seg)
                    feats["LOG"].append(val if np.isfinite(val) else 0)
                except:
                    feats["LOG"].append(0)
                psd, freq = getPSD(seg, fs)
                feats["MNF"].append(getMNF(psd, freq))
                feats["MDF"].append(getMDF(psd, freq))
            txt = ", ".join(f"{k}:{np.mean(v):.2f}" for k, v in feats.items())
            self.chat.log_event(f"Channel {self.selected_channels[idx]+1} Features — {txt}")

    def toggle_recording(self):
        if self.recorder.recording:
            self.recorder.stop_recording()
            self.btn_record.setText("Start Recording")
        else:
            self.recorder.start_recording()
            self.btn_record.setText("Stop Recording")

    def handle_data_update(self, times, values):
        if self.recorder.recording:
            latest_time = times[-1]
            if values.ndim == 1:
                emg_vector = values[-1:]
            else:
                emg_vector = values[-1, :]
            self.recorder.record_data_point(latest_time, emg_vector)


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_M:
            self.recorder.mark_event()
        super().keyPressEvent(event)

    def save_logs(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Logs", "", "Text File (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.chat.get_logs()))
                self.chat.log_event(f"Logs saved to {os.path.basename(path)}")
                QtWidgets.QMessageBox.information(self, "Save Successful", "Logs saved.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def closeEvent(self, event):
        if hasattr(self, 'graph') and self.graph.thread.isRunning():
            self.graph.thread.stop()
        if hasattr(self, 'live'):
            self.live.close()
        self.recorder.close()
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    dialog = StartupDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        mode, data_type = dialog.get_selections()
        mode_label = f"{mode.capitalize()} Mode"
        window = MainWindow(mode=mode, mode_label=mode_label, data_type=data_type)
        window.resize(1400, 800)
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
