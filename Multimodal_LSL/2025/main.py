import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from graph_widget import GraphWidget
from chat_widget import ChatWidget
from emg_features_extractor import EMGFeatureExtractor
from startup_dialog import StartupDialog
from live_mode import LiveMode

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
        title = "PhantasiAi"
        if mode_label:
            title += f" – {mode_label}"
        self.setWindowTitle(title)
        self._build_ui()

        if self.data_type == "live":
            # Real-time mode: tries to connect to Arduino
            self.live = LiveMode(
                port="/dev/ttyUSB0",
                sample_rate=220,
                buffer_seconds=2
            )
            if not self.live.connect():
                QtWidgets.QMessageBox.critical(
                    self, "Connection Error",
                    "Failed to connect to Arduino.\n"
                    "Please make sure it is plugged in and try again."
                )
                sys.exit(1)
            self.setup_live_graph()
        else:
            # Database mode: prompts for .npz file
            if not self.load_npz_file():
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    "No .npz file selected. Closing."
                )
                sys.exit(1)

    def _build_ui(self):
        container = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.btn_open     = QtWidgets.QPushButton("Open")
        self.btn_save     = QtWidgets.QPushButton("Save")
        self.btn_channels = QtWidgets.QPushButton("Channels")
        self.btn_features = QtWidgets.QPushButton("EMG Data")
        btn_size = QtCore.QSize(160, 60)
        font     = QtGui.QFont("Segoe UI", 7)
        for btn in (self.btn_open, self.btn_save,
                    self.btn_channels, self.btn_features):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            btn.setStyleSheet("QPushButton { font-size: 7pt; padding: 7px; }")

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(12, 12, 12, 6)
        toolbar.setSpacing(8)
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_save)
        toolbar.addWidget(self.btn_channels)
        toolbar.addWidget(self.btn_features)
        toolbar.addStretch()
        self.main_layout.addLayout(toolbar)

        self.body_layout = QtWidgets.QHBoxLayout()
        self.body_layout.setContentsMargins(12, 6, 12, 12)
        self.body_layout.setSpacing(10)

        self.chat = ChatWidget(mode=self.mode)
        self.body_layout.addWidget(self.chat)
        self.main_layout.addLayout(self.body_layout)

        self.setCentralWidget(container)

        self.btn_open.clicked.connect(self.on_file_change)
        self.btn_save.clicked.connect(self.save_logs)
        self.btn_channels.clicked.connect(self.show_channel_dialog)
        self.btn_features.clicked.connect(self.compute_features)

    def load_npz_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a .npz file", "", "NumPy Archive (*.npz)"
        )
        if not path:
            return False
        try:
            with np.load(path) as archive:
                key = 'emg' if 'emg' in archive.files else archive.files[0]
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
                w.thread.stop()
                w.setParent(None)

        mat = self.full_data[:, self.selected_channels]
        self.num_channels = len(self.selected_channels)
        self.data_index   = 0

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
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Data paused"))
        self.graph.repris.connect(lambda: self.chat.log_event("Data resumed"))

        for i in reversed(range(self.body_layout.count())):
            self.body_layout.takeAt(i)
        gs, cs = (4, 1) if self.mode=="pro" else (3, 2)
        self.body_layout.addWidget(self.graph, gs)
        self.body_layout.addWidget(self.chat, cs)
        chans = ', '.join(str(c+1) for c in self.selected_channels)
        self.chat.log_event(f"Graph initialized for channels: {chans}")

    def on_file_change(self):
        if self.data_type=="database" and self.load_npz_file():
            self.chat.log_event("Data source changed")

    def show_channel_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Channels")
        layout = QtWidgets.QVBoxLayout(dialog)

        select_all = QtWidgets.QCheckBox("Select/Deselect All")
        select_all.setChecked(len(self.selected_channels)==self.full_data.shape[1])
        layout.addWidget(select_all)

        grid = QtWidgets.QGridLayout()
        self.checkboxes = []
        total = self.full_data.shape[1]
        for idx in range(total):
            cb = QtWidgets.QCheckBox(f"Channel {idx+1}")
            cb.setChecked(idx in self.selected_channels)
            self.checkboxes.append(cb)
            grid.addWidget(cb, idx//4, idx%4)
        layout.addLayout(grid)

        def toggle_all(state):
            for cb in self.checkboxes:
                cb.blockSignals(True)
                cb.setChecked(bool(state))
                cb.blockSignals(False)

        def update_all():
            all_checked = all(cb.isChecked() for cb in self.checkboxes)
            select_all.blockSignals(True)
            select_all.setChecked(all_checked)
            select_all.blockSignals(False)

        select_all.stateChanged.connect(toggle_all)
        for cb in self.checkboxes:
            cb.stateChanged.connect(update_all)

        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn     = QtWidgets.QPushButton("OK")
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
                chans = ', '.join(str(i+1) for i in sel)
                self.chat.log_event(f"Channels selected: {chans}")
                self.build_database_view()

    def compute_features(self):
        mat = self.full_data[:, self.selected_channels]
        extractor = EMGFeatureExtractor(emg_array=mat)
        feats = extractor.features_dict
        for idx, ch in enumerate(self.selected_channels):
            mav = feats['mav'][idx]
            rms = feats['rms'][idx]
            ssc = feats['ssc'][idx]
            var = feats['var'][idx]
            self.chat.log_event(
                f"Channel {ch+1} Features — MAV: {mav:.6f}, "
                f"RMS: {rms:.6f}, SSC: {ssc}, VAR: {var:.6e}"
            )

    def setup_live_graph(self):
        # Clear body
        for i in reversed(range(self.body_layout.count())):
            w = self.body_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        labels = [f"Channel {i+1}" for i in range(len(self.live.channels))]
        self.graph = GraphWidget(
            read_func=self.live.read_latest,
            num_channels=len(self.live.channels),
            channel_labels=labels,
            title="Live EMG",
            sample_rate=self.live.sample_rate,
            buffer_seconds=self.live.buffer_seconds
        )
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Stream paused"))
        self.graph.repris.connect(lambda: self.chat.log_event("Stream resumed"))

        self.body_layout.addWidget(self.graph, 4)
        self.body_layout.addWidget(self.chat, 1)
        self.chat.log_event("Live mode started. Streaming from Arduino.")

    def save_logs(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Logs", "", "Text File (*.txt>"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    f.write("\n".join(self.chat.get_logs()))
                self.chat.log_event(f"Logs saved to {os.path.basename(path)}")
                QtWidgets.QMessageBox.information(
                    self, "Save Successful",
                    f"Logs saved to {os.path.basename(path)}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def closeEvent(self, event):
        if hasattr(self, 'graph') and self.graph.thread.isRunning():
            self.graph.thread.stop()
        if hasattr(self, 'live'):
            self.live.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    dialog = StartupDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        mode, data_type = dialog.get_selections()  
        mode_label = f"{mode.capitalize()} Mode"
        window = MainWindow(
            mode=mode,
            mode_label=mode_label,
            data_type=data_type
        )
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
