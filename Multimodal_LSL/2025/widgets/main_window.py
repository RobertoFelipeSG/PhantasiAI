import os
import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from widgets.graph_widget import GraphWidget
from widgets.chat_widget import ChatWidget
from data.real_time_data import RealTimeData
from emg.emg_recorder import EMGRecorder
from config.config_manager import load_config, save_config

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for PhantasiAI.
    Handles UI, live/file mode switching, EMG recording,
    and centralizes all enable/disable logic.
    """
    def __init__(self, mode="chat", mode_label="", data_type="live", file_path=None):
        super().__init__()
        # Basic setup

    

        self.mode = mode
        self.data_type = data_type
        self.file_path = file_path

        self.config = load_config()
        self.arduino_port = self.config["arduino_port"]
        self.view_mode = self.config.get("view_mode", "chat")

        # State
        self.current_data_source = data_type
        self.selected_channels = []
        self.data_received = False

        self.setWindowTitle("PhantasiAI")
        icon_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),  # this file’s dir
                        os.pardir,                   # “..”
                        "assets",                    # assets folder
                        "favicon.ico"                # your icon
            )
        )
        self.setWindowIcon(QtGui.QIcon(icon_path))

        # Build UI and recorder
        self._build_ui()
        self.recorder = EMGRecorder(self)

        # Kick off in appropriate mode
        if self.data_type == "live":
            self._init_live()
        elif self.data_type == "database":
            if not self.file_path or not self.load_npz_file(self.file_path):
                self.chat.log_event("No .npz file provided.")
                self.show_dummy_graph()

    def _build_ui(self):
        """Create toolbar, layouts, and chat widget."""
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        self.main_layout = QtWidgets.QVBoxLayout(container)

        # Toolbar buttons
        self.btn_open          = QtWidgets.QPushButton("Open")
        self.btn_save          = QtWidgets.QPushButton("Save Chat")
        self.btn_channels      = QtWidgets.QPushButton("Channels")
        self.btn_record        = QtWidgets.QPushButton("Start Recording")
        self.btn_toggle_view   = QtWidgets.QPushButton("Switch to Graph Mode")
        self.btn_retry_arduino = QtWidgets.QPushButton("Reconnect")
        self.btn_spike_mode    = QtWidgets.QPushButton("Disable Auto Markers")
        
        # Create countdown label for top toolbar
        self.countdown_label = QtWidgets.QLabel("Next marker: 5s")
        self.countdown_label.setStyleSheet("QLabel { color: red; font-weight: bold; font-size: 11pt; }")
        self.countdown_label.setAlignment(QtCore.Qt.AlignCenter)

        for btn in (
            self.btn_open, self.btn_save, self.btn_channels,
            self.btn_record, self.btn_toggle_view, self.btn_retry_arduino, self.btn_spike_mode
        ):
            btn.setFixedSize(QtCore.QSize(160, 60))
            btn.setFont(QtGui.QFont("Segoe UI", 7))
            btn.setStyleSheet("QPushButton { font-size: 7pt; padding: 7px; }")

        # Layout toolbar
        toolbar = QtWidgets.QHBoxLayout()
        for btn in (
            self.btn_open, self.btn_save, self.btn_channels,
            self.btn_record, self.btn_toggle_view, self.btn_retry_arduino, self.btn_spike_mode
        ):
            toolbar.addWidget(btn)
        toolbar.addStretch()
        toolbar.addWidget(self.countdown_label)  # Add countdown to the right
        self.main_layout.addLayout(toolbar)

        # Body: chat + graph area
        self.body_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.body_layout)

        self.chat = ChatWidget(mode=self.mode)
        self.body_layout.addWidget(self.chat, 1)

        # Connect signals
        self.btn_open.clicked.connect(self.on_file_change)
        self.btn_save.clicked.connect(self.save_logs)
        self.btn_channels.clicked.connect(self.show_channel_dialog)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_toggle_view.clicked.connect(self.toggle_display_mode)
        self.btn_retry_arduino.clicked.connect(self.retry_arduino_connection)
        self.btn_spike_mode.clicked.connect(self.toggle_spike_mode)

    def _init_live(self):
        """Attempt Arduino live connection or fallback to dummy graph."""
        self.live = RealTimeData(port=self.arduino_port, sample_rate=220, buffer_seconds=2)
        if not self.live.connect():
            self.chat.log_event("Sensor not detected. You can still open a file.")
            self.current_data_source = "database"
            self.show_dummy_graph()
            return

        self.chat.set_mode("live")
        self.chat.log_event("Sensor detected. Live mode started.")
        self.selected_channels = [0]
        self.setup_live_graph()

    def show_dummy_graph(self):
        """Display placeholder graph and disable real-data controls."""
        self._remove_existing_graph()
        self.graph = GraphWidget(
            read_func=lambda: [0.0],
            num_channels=1,
            channel_labels=["No Data"],
            title="Arduino Not Detected",
            sample_rate=220,
            buffer_seconds=2
        )
        self.graph.dummy_mode = True

        self.chat.set_mode("live")
        self.body_layout.insertWidget(0, self.graph, 4)
        self._apply_view_mode()
        self._update_controls()

    def retry_arduino_connection(self):
        """Reconnect handler for live mode."""
        self.chat.log_event("Attempting to reconnect...")
        new_live = RealTimeData(port=self.arduino_port, sample_rate=220, buffer_seconds=2)
        if new_live.connect():
            self.live = new_live
            self.current_data_source = "live"
            self.data_type = "live"
            self.selected_channels = [0]
            self.chat.set_mode("live")
            self.setup_live_graph()
            self.chat.log_event("Reconnected successfully.")
        else:
            self.chat.log_event("Reconnect Failed: Sensor still not detected.")
        self._update_controls()

    def setup_live_graph(self):
        """Build live-EMG graph, start streaming, and update controls."""
        self._remove_existing_graph()
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
        # Connect data update only; omit pause/resume logging
        self.graph.thread.dataUpdated.connect(self.handle_data_update)
        self.graph.countdown_updated.connect(self.update_countdown)

        self.body_layout.insertWidget(0, self.graph, 4)
        self._apply_view_mode()
        self._update_controls()

    def load_npz_file(self, path=None):
        """Load EMG data from a .npz file into database mode."""
        if not path:
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
            self.chat.set_mode("file")
            self.chat.log_event(f"Opened file «{os.path.basename(path)}».")
            self.build_database_view()
            return True
        except Exception as e:
            self.chat.log_event(f"Error loading .npz: {e}")
            return False

    def build_database_view(self):
        """Play back file data in a looping graph and update controls."""
        self._remove_existing_graph()

        mat = self.full_data[:, self.selected_channels]
        self.data_index = 0

        def read_vector():
            v = mat[self.data_index]
            self.data_index = (self.data_index + 1) % len(mat)
            return v.tolist()

        labels = [f"Channel {c+1}" for c in self.selected_channels]
        self.graph = GraphWidget(
            read_func=read_vector,
            num_channels=len(self.selected_channels),
            channel_labels=labels,
            title="EMG Channels",
            sample_rate=self.sample_rate,
            buffer_seconds=self.buffer_seconds
        )
        self.graph.thread.dataUpdated.connect(self.handle_data_update)
        self.graph.countdown_updated.connect(self.update_countdown)

        self.body_layout.insertWidget(0, self.graph, 4)
        self._apply_view_mode()
        self.chat.log_event(
            f"Graph initialized for channels: {', '.join(str(c+1) for c in self.selected_channels)}"
        )
        self._update_controls()

    def _remove_existing_graph(self):
        """Remove existing GraphWidget and stop its thread."""
        for i in reversed(range(self.body_layout.count())):
            w = self.body_layout.itemAt(i).widget()
            if isinstance(w, GraphWidget):
                if w.thread.isRunning():
                    w.thread.stop()
                w.setParent(None)
                w.deleteLater()

    def _apply_view_mode(self):
        """Show or hide the graph based on the saved view_mode."""
        if self.view_mode == "graph":
            self.graph.show()
            self.btn_toggle_view.setText("Chat Mode")
        else:
            self.graph.hide()
            self.btn_toggle_view.setText("Graph Mode")

    def toggle_display_mode(self):
        """Switch between chat and graph views, persist setting, and log."""
        if not hasattr(self, 'graph'):
            self.chat.log_event("Graph not available yet; cannot switch view.")
            return
        if self.graph.isVisible():
            self.graph.hide()
            self.btn_toggle_view.setText("Graph Mode")
            self.chat.log_event("Switched to Chat Mode")
            self.view_mode = "chat"
        else:
            self.graph.show()
            self.btn_toggle_view.setText("Chat Mode")
            self.chat.log_event("Switched to Graph Mode")
            self.view_mode = "graph"
        self.config["view_mode"] = self.view_mode
        save_config(self.config)

    def on_file_change(self):
        """Handler for 'Open' button."""
        if self.load_npz_file():
            self.current_data_source = "database"

    def show_channel_dialog(self):
        """Original modal channel selector (for fallback)."""
        total = len(self.live.channels) if self.current_data_source == "live" else self.full_data.shape[1]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Channels")
        layout = QtWidgets.QVBoxLayout(dialog)

        select_all = QtWidgets.QCheckBox("Select/Deselect All")
        layout.addWidget(select_all)

        grid = QtWidgets.QGridLayout()
        self.checkboxes = []
        for idx in range(total):
            cb = QtWidgets.QCheckBox(f"Channel {idx+1}")
            cb.setChecked(idx in self.selected_channels)
            self.checkboxes.append(cb)
            grid.addWidget(cb, idx // 4, idx % 4)
        layout.addLayout(grid)

        def toggle_all(state):
            for cb in self.checkboxes:
                cb.setChecked(bool(state))

        def update_all():
            select_all.setChecked(all(cb.isChecked() for cb in self.checkboxes))

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
                self.chat.log_event(f"Channels selected: {', '.join(str(i+1) for i in sel)}")
                if self.current_data_source == "live":
                    self.setup_live_graph()
                else:
                    self.build_database_view()

    def toggle_recording(self):
        """
        Start or stop EMG recording.
        Only allowed when real data is connected (live or file).
        """
        if not hasattr(self, 'graph') or getattr(self.graph, 'dummy_mode', False):
            self.chat.log_event("Recording disabled: no real data source.")
            return

        if self.current_data_source == "live" and not self.data_received:
            self.chat.log_event("Recording disabled: waiting for live data.")
            return

        if self.recorder.recording:
            self.recorder.stop_recording()
            self.btn_record.setText("Start Recording")
            
            if self.recorder.session_dir:
                folder = self.recorder.get_session_folder_name()
                self.chat.log_event(f"Recording saved in folder «{folder}»")

        else:
            self.recorder.start_recording()
            self.btn_record.setText("Stop Recording")
            self.chat.log_event("Recording started.")
            

        # Refresh button states
        self._update_controls()
     


    def handle_data_update(self, times, values):
        """Callback on new data; enables recording after first packet."""
        if not self.data_received:
            self.data_received = True
            self._update_controls()
        if self.recorder.recording:
            latest_time = times[-1]
            emg_vector = values[-1, :] if values.ndim > 1 else [values[-1]]
            self.recorder.record_data_point(latest_time, emg_vector)

    def update_countdown(self, countdown_text):
        """Update the countdown label in the top toolbar."""
        self.countdown_label.setText(countdown_text)
    
    def toggle_spike_mode(self):
        """Toggle automatic spike markers on/off."""
        if not hasattr(self, 'graph') or not self.graph:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please initialize a graph first.")
            return
        
        if self.graph.auto_spike_enabled:
            # Disable auto spikes
            self.graph.set_auto_spike_enabled(False)
            self.btn_spike_mode.setText("Enable Auto Markers")
            self.chat.log_event("Automatic markers disabled")
        else:
            # Enable auto spikes - first ask for interval
            interval, ok = QtWidgets.QInputDialog.getDouble(
                self, 
                "Set Marker Interval", 
                "Enter interval between markers (seconds):",
                value=self.graph.spike_interval,
                min=0.1,
                max=60.0,
                decimals=1
            )
            
            if ok:
                self.graph.set_spike_interval(interval)
                self.graph.set_auto_spike_enabled(True)
                self.btn_spike_mode.setText("Disable Auto Markers")
                self.chat.log_event(f"Automatic markers enabled (every {interval}s)")
            else:
                # User cancelled, don't change state
                return

    def keyPressEvent(self, event):
        """Press 'M' to add a marker at latest timestamp."""
        if event.key() == QtCore.Qt.Key_M and hasattr(self, 'graph'):
            if self.graph.curves and self.graph.curves[0].getData()[0].size > 0:
                t_last = self.graph.curves[0].getData()[0][-1]
                self.graph.add_marker(t_last)
                self.chat.log_event(f"🟡 Marker added at {t_last:.2f} s")
                self.recorder.mark_event()
        super().keyPressEvent(event)

    def save_logs(self):
        """Save chat logs to a text file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Logs", "", "Text File (*.txt)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.chat.get_logs()))
                self.chat.log_event(f"Logs saved to {os.path.basename(path)}")
            except Exception as e:
                self.chat.log_event(f"Save Error: {e}")

    def closeEvent(self, event):
        """Clean up threads/connections on close."""
        if hasattr(self, 'graph') and self.graph.thread.isRunning():
            self.graph.thread.stop()
        if hasattr(self, 'live'):
            self.live.close()
        self.recorder.close()
        super().closeEvent(event)

    def _update_controls(self):
        """
        Centralized enable/disable logic for toolbar + graph controls.
        Disables pause/zoom when dummy_mode is True.
        """
        dummy = getattr(self, 'graph', None) and getattr(self.graph, 'dummy_mode', False)

        self.btn_open.setEnabled(True)
        self.btn_save.setEnabled(True)

        self.btn_channels.setEnabled(not dummy)

        self.btn_record.setEnabled(not dummy and self.data_received)

        self.btn_toggle_view.setEnabled(hasattr(self, 'graph'))

        self.btn_retry_arduino.setEnabled(dummy and self.data_type == "live")

        if hasattr(self.graph, 'btn_pause'):
            self.graph.btn_pause.setEnabled(not dummy)
        if hasattr(self.graph, 'btn_zoom_in'):
            self.graph.btn_zoom_in.setEnabled(not dummy)
        if hasattr(self.graph, 'btn_zoom_out'):
            self.graph.btn_zoom_out.setEnabled(not dummy)
        if hasattr(self.graph, 'btn_reset_zoom'):
            self.graph.btn_reset_zoom.setEnabled(not dummy)
