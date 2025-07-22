# main_window.py

import os
import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui, QtCore

# Internal module imports
from widgets.graph_widget import GraphWidget
from widgets.chat_widget import ChatWidget
from data.live_mode import LiveMode
from emg.emg_recorder import EMGRecorder

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PhantasiAi GUI.
    Handles UI setup, file I/O, EMG data streaming, channel selection, and recording.
    """

    def __init__(self, mode, mode_label="", data_type="database"):
        """
        Initialize the main window with specified mode and data type.

        Args:
            mode: UI mode ('pro' or 'chat')
            mode_label: Label appended to window title
            data_type: 'live' (real-time Arduino) or 'database' (.npz file)
        """
        super().__init__()
        self.mode = mode
        self.data_type = data_type
        self.selected_channels = []

        # Set the window title
        title = "PhantasiAi" + (f" – {mode_label}" if mode_label else "")
        self.setWindowTitle(title)

        # Build interface components
        self._build_ui()

        # Initialize the EMG recording handler
        self.recorder = EMGRecorder(self)

        # Load data source (live or database)
        if self.data_type == "live":
            self._init_live()
        else:
            if not self.load_npz_file():
                QtWidgets.QMessageBox.critical(self, "Error", "No .npz file selected. Closing.")
                sys.exit(1)

    def _build_ui(self):
        """
        Create the main layout, toolbar buttons, chat panel, and event connections.
        """
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        self.main_layout = QtWidgets.QVBoxLayout(container)

        # Toolbar with buttons
        self.btn_open     = QtWidgets.QPushButton("Open")
        self.btn_save     = QtWidgets.QPushButton("Save Chat")
        self.btn_channels = QtWidgets.QPushButton("Channels")
        self.btn_record   = QtWidgets.QPushButton("Start Recording")

        # Standard styling for all toolbar buttons
        for btn in (self.btn_open, self.btn_save, self.btn_channels, self.btn_record):
            btn.setFixedSize(QtCore.QSize(160, 60))
            btn.setFont(QtGui.QFont("Segoe UI", 7))
            btn.setStyleSheet("QPushButton { font-size: 7pt; padding: 7px; }")

        # Toolbar layout
        toolbar = QtWidgets.QHBoxLayout()
        for btn in (self.btn_open, self.btn_save, self.btn_channels, self.btn_record):
            toolbar.addWidget(btn)
        toolbar.addStretch()
        self.main_layout.addLayout(toolbar)

        # Main horizontal layout: Chat + Graph
        self.body_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.body_layout)

        # Add chat panel
        self.chat = ChatWidget(mode=self.mode)
        self.body_layout.addWidget(self.chat)

        # Button connections
        self.btn_open.clicked.connect(self.on_file_change)
        self.btn_save.clicked.connect(self.save_logs)
        self.btn_channels.clicked.connect(self.show_channel_dialog)
        self.btn_record.clicked.connect(self.toggle_recording)

    def _init_live(self):
        """
        Initialize the Arduino connection and start live data stream.
        """
        self.live = LiveMode(port="COM3", sample_rate=220, buffer_seconds=2)
        if not self.live.connect():
            QtWidgets.QMessageBox.critical(
                self, "Connection Error",
                "Failed to connect to Arduino.\nPlease make sure it is plugged in and try again."
            )
            sys.exit(1)

        self.selected_channels = [0]  # Default to first channel
        self.setup_live_graph()

    def load_npz_file(self):
        """
        Load EMG data from a .npz file. Prompt user to select a file.
        """
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
        """
        Set up a graph for offline .npz data using selected channels.
        """
        # Remove any existing graph widget
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

        # Create a read function that loops through data
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

        # Connect graph signals to logging and recording
        self.graph.thread.dataUpdated.connect(self.handle_data_update)
        self.graph.paused.connect(lambda: self.chat.log_event("Data paused"))
        self.graph.resumed.connect(lambda: self.chat.log_event("Data resumed"))

        # Layout placement
        if self.mode == "pro":
            self.body_layout.addWidget(self.graph, 4)
            self.body_layout.addWidget(self.chat, 1)
        else:
            self.body_layout.addWidget(self.chat, 1)

        self.chat.log_event(f"Graph initialized for channels: {', '.join(str(c+1) for c in self.selected_channels)}")

    def setup_live_graph(self):
        """
        Initialize real-time graph using Arduino EMG stream.
        """
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
        self.graph.paused.connect(lambda: self.chat.log_event("Stream paused"))
        self.graph.resumed.connect(lambda: self.chat.log_event("Stream resumed"))

        if self.mode == "pro":
            self.body_layout.addWidget(self.graph, 4)
            self.body_layout.addWidget(self.chat, 1)
        else:
            self.body_layout.addWidget(self.chat, 1)

        self.chat.log_event("Live mode started. Streaming from Arduino.")

    def on_file_change(self):
        """
        Reload .npz file when the 'Open' button is clicked.
        """
        if self.data_type == "database" and self.load_npz_file():
            self.chat.log_event("Data source changed")

    def show_channel_dialog(self):
        """
        Open a dialog allowing the user to choose which EMG channels to display/record.
        """
        total_channels = len(self.live.channels) if self.data_type == "live" else self.full_data.shape[1]

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Channels")
        layout = QtWidgets.QVBoxLayout(dialog)

        # Select-all option
        select_all = QtWidgets.QCheckBox("Select/Deselect All")
        layout.addWidget(select_all)

        # Channel checkboxes
        grid = QtWidgets.QGridLayout()
        self.checkboxes = []
        for idx in range(total_channels):
            cb = QtWidgets.QCheckBox(f"Channel {idx+1}")
            cb.setChecked(idx in self.selected_channels)
            self.checkboxes.append(cb)
            grid.addWidget(cb, idx // 4, idx % 4)
        layout.addLayout(grid)

        # Sync "select all" checkbox with individual ones
        def toggle_all(state): [cb.setChecked(bool(state)) for cb in self.checkboxes]
        def update_all():
            all_checked = all(cb.isChecked() for cb in self.checkboxes)
            select_all.setChecked(all_checked)

        select_all.stateChanged.connect(toggle_all)
        for cb in self.checkboxes:
            cb.stateChanged.connect(update_all)

        # OK/Cancel buttons
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # Apply selection if user confirmed
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            sel = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
            if sel:
                self.selected_channels = sel
                self.chat.log_event(f"Channels selected: {', '.join(str(i+1) for i in sel)}")
                if self.data_type == "live":
                    self.setup_live_graph()
                else:
                    self.build_database_view()

    def toggle_recording(self):
        """
        Start or stop the EMG recording session.
        """
        if self.recorder.recording:
            self.recorder.stop_recording()
            self.btn_record.setText("Start Recording")
        else:
            self.recorder.start_recording()
            self.btn_record.setText("Stop Recording")

    def handle_data_update(self, times, values):
        """
        Receive EMG data updates and store them in the recording file if recording is active.
        """
        if self.recorder.recording:
            latest_time = times[-1]
            emg_vector = values[-1, :] if values.ndim > 1 else [values[-1]]
            self.recorder.record_data_point(latest_time, emg_vector)

    def keyPressEvent(self, event):
        """
        Keyboard shortcut: Press 'M' to mark a timestamped event during recording.
        """
        if event.key() == QtCore.Qt.Key_M:
            self.recorder.mark_event()
        super().keyPressEvent(event)

    def save_logs(self):
        """
        Save the chat/event logs to a .txt file.
        """
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
        """
        Clean up threads and connections before the window closes.
        """
        if hasattr(self, 'graph') and self.graph.thread.isRunning():
            self.graph.thread.stop()
        if hasattr(self, 'live'):
            self.live.close()
        self.recorder.close()
        super().closeEvent(event)
