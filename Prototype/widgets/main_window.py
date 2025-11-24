import os
import sys
import subprocess
import threading
import queue
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from widgets.graph_widget import GraphWidget
from widgets.chat_widget import ChatWidget
from data.real_time_data import RealTimeData
from data.ganglion_data import GanglionData
from emg.real_time.real_time_recorder import RealTimeRecorder
from config.config_manager import load_config, save_config

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for PhantasiAI.
    Handles UI, live/file mode switching, EMG recording,
    and centralizes all enable/disable logic.
    """
    def __init__(self, mode="chat", mode_label="", data_type="live", file_path=None, time_interval=60):
        super().__init__()
        # Basic setup

    
        self.time_interval=time_interval
        self.mode = mode
        self.data_type = data_type
        self.file_path = file_path

        self.config = load_config()

        # Prefer config, but auto-probe sane defaults on Pi if missing/empty
        cfg_port = (self.config.get("arduino_port") or "").strip()
        probed_ports = [p for p in ("/dev/ttyUSB0", "/dev/ttyACM0") if os.path.exists(p)]
        self.arduino_port = cfg_port or (probed_ports[0] if probed_ports else "/dev/ttyUSB0")
        
        # ganglion port configuration
        cfg_ganglion_port = (self.config.get("ganglion_port") or "").strip()
        probed_ports = [p for p in ("/dev/ttyUSB0", "/dev/ttyACM0") if os.path.exists(p)]
        self.ganglion_port = cfg_ganglion_port or (probed_ports[0] if probed_ports else "/dev/ttyACM0")
        
        # sensor type selection
        self.sensor_type = self.config.get("sensor_type","ganglion") # "arduino" or "ganglion"

        self.view_mode = self.config.get("view_mode", "chat")


        # State
        self.current_data_source = data_type
        self.selected_channels = []
        self.data_received = False
        
        # Background processes for AI and STIM
        self.ai_process = None
        self.stim_process = None
        self.ai_output_queue = queue.Queue()
        self.stim_output_queue = queue.Queue()
        self.output_timer = QtCore.QTimer()
        self.output_timer.timeout.connect(self.process_background_output)

        # Automatic analysis settings
        self.auto_analysis_enabled = False
        self.analysis_interval = self.time_interval  # 60 seconds = 1 minute
        self.auto_analysis_timer = QtCore.QTimer()
        self.auto_analysis_timer.timeout.connect(self.perform_automatic_analysis)
        self.analysis_countdown = self.time_interval
        self.countdown_timer = QtCore.QTimer()
        self.countdown_timer.timeout.connect(self.update_analysis_countdown)

        self.setWindowTitle("PhantasiAI")
        icon_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),  # this file's dir
                        os.pardir,                   # ".."
                        "assets",                    # assets folder
                        "favicon.ico"                # your icon
            )
        )
        self.setWindowIcon(QtGui.QIcon(icon_path))

        # Build UI and recorder
        self._build_ui()
        self.recorder = RealTimeRecorder(self)

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
        self.btn_save          = QtWidgets.QPushButton("Save log")
        self.btn_channels      = QtWidgets.QPushButton("Channel")
        self.btn_record        = QtWidgets.QPushButton("START")
        self.btn_toggle_view   = QtWidgets.QPushButton("Graph Mode")
        self.btn_retry_arduino = QtWidgets.QPushButton("Connect")
        self.btn_sensor_type   = QtWidgets.QPushButton("Ganglion" if self.sensor_type == "arduino" else "Arduino")
        self.btn_spike_mode    = QtWidgets.QPushButton("Markers")
        self.btn_auto_analysis = QtWidgets.QPushButton("Analysis")
        
        # Create countdown label for top toolbar
        self.countdown_label = QtWidgets.QLabel("Marker: 5s")
        self.countdown_label.setStyleSheet("QLabel { color: red; font-weight: bold; font-size: 15pt; }")
        self.countdown_label.setAlignment(QtCore.Qt.AlignCenter)

        # Create auto analysis countdown label
        self.auto_analysis_countdown_label = QtWidgets.QLabel("AI Analysis: OFF")
        self.auto_analysis_countdown_label.setStyleSheet("QLabel { color: blue; font-weight: bold; font-size: 15pt; }")
        self.auto_analysis_countdown_label.setAlignment(QtCore.Qt.AlignCenter)

        for btn in (
            self.btn_open, self.btn_save, self.btn_channels,
            self.btn_record, self.btn_auto_analysis, self.btn_toggle_view, self.btn_retry_arduino, self.btn_sensor_type, self.btn_spike_mode
        ):
            btn.setFixedSize(QtCore.QSize(50, 30)) #160, 60
            btn.setFont(QtGui.QFont("Segoe UI", 2)) #7
            btn.setStyleSheet("QPushButton { font-size: 7pt; padding: 7px; }")

        # Layout toolbar
        toolbar = QtWidgets.QHBoxLayout()
        for btn in (
            self.btn_open, self.btn_save, self.btn_channels,
            self.btn_record, self.btn_auto_analysis, self.btn_toggle_view, self.btn_retry_arduino, self.btn_sensor_type, self.btn_spike_mode
        ):
            toolbar.addWidget(btn)
        toolbar.addStretch()
        toolbar.addWidget(self.countdown_label)  # Add countdown to the right
        toolbar.addWidget(self.auto_analysis_countdown_label)  # Add auto analysis countdown
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
        self.btn_sensor_type.clicked.connect(self.toggle_sensor_type)
        self.btn_spike_mode.clicked.connect(self.toggle_spike_mode)
        self.btn_auto_analysis.clicked.connect(self.toggle_auto_analysis)


    def _init_live(self):
        """Attempt sensor live connection or fallback to dummy graph."""
        if self.sensor_type == "ganglion":
            self._init_ganglion()
        else:
            self._init_arduino()
        
        
        
    def _init_arduino(self):
        """Attempt arduino live connection or fallback to dummy graph."""
        self.chat.log_event(f"Attempting arduino to connect on {self.arduino_port} ...")
        self.live = RealTimeData(port=self.arduino_port, sample_rate=220, buffer_seconds=2)
        if not self.live.connect():
            # Try the alternate common port once
            alt = "/dev/ttyACM0" if self.arduino_port.endswith("USB0") else "/dev/ttyUSB0"
            if os.path.exists(alt):
                self.chat.log_event(f"Primary failed. Trying alternate port {alt} ...")
                alt_live = RealTimeData(port=alt, sample_rate=220, buffer_seconds=2)
                if alt_live.connect():
                    self.live = alt_live
                    self.arduino_port = alt
                else:
                    self.chat.log_event("Alternate port failed as well.")
                    self.current_data_source = "database"
                    self.show_dummy_graph()
                    return
            else:
                self.chat.log_event("Arduino not detected. You can still open a file.")
                self.current_data_source = "database"
                self.show_dummy_graph()
                return


        self.chat.set_mode("live")
        self.chat.log_event("Arduino detected. Live mode started.")
        self.selected_channels = [0]
        self.setup_live_graph()
        
    def _init_ganglion(self):
        """Attempt ganglion live connection or fallback to dummy graph."""
        self.chat.log_event(f"Attempting ganglion to connect on {self.ganglion_port} ...")
        self.live = GanglionData(port=self.ganglion_port, channel_list=[0,1,2,3], sample_rate=200, buffer_seconds=2)
        if not self.live.connect():
            self.chat.log_event("Ganglion not detected. You can still open a file.")
            self.current_data_source = "database"
            self.show_dummy_graph()
            return

        self.chat.set_mode("live")
        self.chat.log_event("Ganglion detected. Live mode started.")
        self.selected_channels = [0]
        self.setup_live_graph()
        

    def show_dummy_graph(self):
        """Display placeholder graph and disable real-data controls."""
        self._remove_existing_graph()
        sensor_name = "Ganglion Board" if self.sensor_type == "ganglion" else "Arduino"
        self.graph = GraphWidget(
            read_func=lambda: [0.0],
            num_channels=1,
            channel_labels=["No Data"],
            title="Sensor Not Detected",
            sample_rate=220,
            buffer_seconds=2
        )
        self.graph.dummy_mode = True
        # enable automatic markers
        self.graph.automatic_marker_added.connect(self.handle_automatic_marker)

        self.chat.set_mode("live")
        self.body_layout.insertWidget(0, self.graph, 4)
        self._apply_view_mode()
        self._update_controls()
        
    
    def retry_arduino_connection(self):
        """Reconnect handler for live mode"""
        if self.sensor_type == "ganglion":
            self._retry_ganglion_connection()
        else:
            self._retry_arduino_connection()
            
    

    def _retry_arduino_connection(self):
        """Reconnect arduino for live mode."""
        self.chat.log_event("Attempting to reconnect Arduino...")
        tried = []
        for port in (self.arduino_port,
                     "/dev/ttyACM0" if self.arduino_port.endswith("USB0") else "/dev/ttyUSB0"):
            if port in tried or not os.path.exists(port):
                continue
            tried.append(port)
            self.chat.log_event(f"Reconnecting Arduino on {port} ...")
            new_live = RealTimeData(port=port, sample_rate=220, buffer_seconds=2)
            if new_live.connect():
                self.live = new_live
                self.arduino_port = port
                self.current_data_source = "live"
                self.data_type = "live"
                self.selected_channels = [0]
                self.chat.set_mode("live")
                self.setup_live_graph()
                self.chat.log_event("Arduino Reconnected successfully.")
                self._update_controls()
                return

        self.chat.log_event("Reconnect Failed: Arduino still not detected.")
        self._update_controls()
        
        
    def _retry_ganglion_connection(self):
        """Reconnect ganglion for live mode."""
        self.chat.log_event("Attempting to reconnect Ganglion...")
        self.chat.log_event(f"Reconnecting Ganglion on {self.ganglion_port} ...")
        new_live = GanglionData(port=self.ganglion_port, channel_list=[0,1,2,3], sample_rate=200, buffer_seconds=2)
        if new_live.connect():
                self.live = new_live
                self.ganglion_port = self.ganglion_port
                self.current_data_source = "live"
                self.data_type = "live"
                self.selected_channels = [0]
                self.chat.set_mode("live")
                self.setup_live_graph()
                self.chat.log_event("Ganglion Reconnected successfully.")
                self._update_controls()
                return
            
        self.chat.log_event("Reconnect Failed: Ganglion still not detected.")
        self._update_controls()    
        
    def toggle_sensor_type(self):
        """ Toggle between arduino or ganglion sensor"""
        if self.sensor_type == "arduino":
            self.sensor_type = "ganglion"
            self.btn_sensor_type.setText("Switch to arduino")
            self.chat.log_event("Switched to ganglion")
        else:
            self.sensor_type = "arduino"
            self.btn_sensor_type.setText("Switch to ganglion")
            self.chat.log_event("Switched to arduino")
        
        #save to config
        self.config["sensor_type"] = self.sensor_type
        save_config(self.config)
        
        if self.current_data_source == "Live":
            self.chat.log_event("Reconnecting with sensor...")
            if hasattr(self, 'Live'):
                self.live.close()
            self._init_live()
            
        

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
        #self.graph.countdown_updated.connect(self.update_countdown)
        
        # Connect synchronized timing signals
        self.graph.thread.analysisRequested.connect(self.perform_automatic_analysis)
        #self.graph.automatic_marker_added.connect(self.handle_automatic_marker)

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
        #self.graph.countdown_updated.connect(self.update_countdown)
        
        # Connect synchronized timing signals
        self.graph.thread.analysisRequested.connect(self.perform_automatic_analysis)
        #self.graph.automatic_marker_added.connect(self.handle_automatic_marker)

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
            self.btn_toggle_view.setText("Chat")
        else:
            self.graph.hide()
            self.btn_toggle_view.setText("Graph")

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
            # Stop auto analysis timer if running
            if self.auto_analysis_enabled:
                self.stop_auto_analysis_timer()
                self.auto_analysis_enabled = False
                self.btn_auto_analysis.setText("Enable Auto Analysis")
                self.auto_analysis_countdown_label.setText("AI analysis: OFF")
                self.auto_analysis_countdown_label.setStyleSheet("QLabel { color: blue; font-weight: bold; font-size: 15pt; }")
            
            # Stop background processes
            self.stop_background_processes()
            self.output_timer.stop()
            self.chat.log_event("Background AI and STIM processes stopped")
            
            self.recorder.stop_recording()
            self.btn_record.setText("START")
            
            if self.recorder.session_dir:
                self.chat.log_event(f"Recording saved")

        else:
            # Show marker interval dialog before starting recording
            if not self.show_marker_interval_dialog():
                return  # User cancelled, don't start recording
            
            self.recorder.start_recording()
            self.btn_record.setText("STOP")
            self.chat.log_event("Recording started.")
            
            # Automatically start "real-time" analysis
            self.auto_analysis_enabled = True
            self.start_auto_analysis_timer()
            self.chat.log_event("Automatic analysis enabled")
            
                                   
            # Start automatically the event timer and markers
            self.graph.countdown_updated.connect(self.update_countdown)
            self.graph.automatic_marker_added.connect(self.handle_automatic_marker)
            

            # Start background processes for AI and STIM
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            script_path = os.path.join(base_dir, "stim", "detect_change.py")
            script_path_II = os.path.join(base_dir, "stim", "detect_2nd_change.py")
            
            # Start AI process in background
            self.chat.log_event(f"Starting AI process: {script_path}")
            self.ai_process = self.start_background_process(script_path, "AI", self.ai_output_queue)
            if self.ai_process:
                self.chat.log_event("AI Stimulation Optimization started in background")
                print("AI process started in background")
            else:
                self.chat.log_event("Failed to start AI process")
                print("Failed to start AI process")
            
            # Start STIM process in background
            self.chat.log_event(f"Starting STIM process: {script_path_II}")
            self.stim_process = self.start_background_process(script_path_II, "STIM", self.stim_output_queue)
            if self.stim_process:
                self.chat.log_event("AI Stimulation started in background")
                print("STIM process started in background")
            else:
                self.chat.log_event("Failed to start STIM process")
                print("Failed to start STIM process")
            
            # Start output processing timer
            self.output_timer.start(100)  # Check for output every 100ms
            
            

        # Refresh button states
        self._update_controls()

    def toggle_auto_analysis(self):
        """
        Toggle automatic 1-minute analysis on/off.
        """
        if not self.recorder.recording:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please start recording first before enabling auto analysis.")
            return
        
        if self.auto_analysis_enabled:
            # Disable auto analysis
            self.stop_auto_analysis_timer()
            self.auto_analysis_enabled = False
            self.btn_auto_analysis.setText("Enable Auto Analysis")
            self.auto_analysis_countdown_label.setText("AI analysis: OFF")
            self.auto_analysis_countdown_label.setStyleSheet("QLabel { color: blue; font-weight: bold; font-size: 15pt; }")
            self.chat.log_event("Automatic analysis disabled")
        else:
            # Enable auto analysis
            self.auto_analysis_enabled = True
            self.btn_auto_analysis.setText("Disable Auto Analysis")
            self.auto_analysis_countdown_label.setStyleSheet("QLabel { color: green; font-weight: bold; font-size: 15pt; }")
            self.chat.log_event(f"AI analysis enabled (every {self.analysis_interval} seconds)")
            self.start_auto_analysis_timer()

    def start_auto_analysis_timer(self):
        """Start the automatic analysis timer."""
        if self.auto_analysis_enabled and self.recorder.recording:
            # Use DataThread for synchronized timing
            if hasattr(self, 'graph') and hasattr(self.graph, 'thread'):
                self.graph.thread.set_analysis_interval(self.analysis_interval)
                self.graph.thread.set_auto_analysis_enabled(True)
            
            # Start countdown timer for display only
            self.countdown_timer.start(1000)  # Update countdown every second
            self.analysis_countdown = self.analysis_interval
            self.update_analysis_countdown()

    def stop_auto_analysis_timer(self):
        """Stop the automatic analysis timer."""
        # Disable DataThread auto analysis
        if hasattr(self, 'graph') and hasattr(self.graph, 'thread'):
            self.graph.thread.set_auto_analysis_enabled(False)
        
        # Stop countdown timer
        self.countdown_timer.stop()
        self.auto_analysis_countdown_label.setText("AI analysis: OFF")

    def update_analysis_countdown(self):
        """Update the analysis countdown display."""
        if self.auto_analysis_enabled and self.recorder.recording:
            # Use DataThread timing for accurate countdown
            if hasattr(self, 'graph') and hasattr(self.graph, 'thread') and self.graph.thread.isRunning():
                current_time = self.graph.thread.get_current_time()
                time_since_last = current_time - getattr(self.graph.thread, 'last_analysis_time', 0)
                time_until_next = self.analysis_interval - time_since_last
                
                if time_until_next > 0:
                    minutes = int(time_until_next) // 60
                    seconds = int(time_until_next) % 60
                    self.auto_analysis_countdown_label.setText(f"AI analysis: {minutes:02d}:{seconds:02d}")
                else:
                    self.auto_analysis_countdown_label.setText("AI analysis: 00:00")
            else:
                # Fallback to old method if DataThread not available
                self.analysis_countdown -= 1
                if self.analysis_countdown <= 0:
                    self.analysis_countdown = self.analysis_interval
                
                minutes = self.analysis_countdown // 60
                seconds = self.analysis_countdown % 60
                self.auto_analysis_countdown_label.setText(f"AI analysis: {minutes:02d}:{seconds:02d}")

    def perform_automatic_analysis(self, timestamp=None):
        """
        Perform automatic peak detection from a temporary file stored in /temp wiht the last minute of data.
        Called by DataThread when it's time for analysis (synchronized timing).
        """
        if not self.recorder.recording or not self.auto_analysis_enabled:
            return
        
        try:
            self.chat.log_event("Performing automatic analysis...")
            
            # Create a temporary file with the last minute of data
            temp_filename = self.recorder.create_temp_analysis_file(self.time_interval)
            
            if temp_filename and os.path.exists(temp_filename):
                # Perform peak analysis and classification
                from emg.offline_analysis.peak_classifier import PeakClassifier
                

                analyzer = PeakClassifier(csv_path=temp_filename)
                results = analyzer.run(show_plots=False, save_results=True, classify_peaks=True)
                
                # Log results
                if results.get('classifications'):
                    num_classifications = len(results['classifications'])
                    self.chat.log_event(f"Auto analysis: {results['num_peaks']} peaks detected, {num_classifications} classified")
                    
                    # Show classification summary
                    class_counts = {}
                    for result in results['classifications']:
                        cls = result['predicted_class']
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    
                    summary = ", ".join([f"{cls}% MVC: {count}" for cls, count in sorted(class_counts.items())])
                    self.chat.log_event(f"Classification: {summary}")
                else:
                    self.chat.log_event(f"Auto analysis: {results['num_peaks']} peaks detected (no classification)")
                
                # Clean up temporary file
                #try:
                #    os.remove(temp_filename)
                #except:
                #    pass
            else:
                self.chat.log_event("Auto analysis: No data available for analysis")
                
        except Exception as e:
            self.chat.log_event(f"Auto analysis error: {e}")
            print(f"[Auto Analysis] Error: {e}")
     


    def handle_data_update(self, times, values):
        """Callback on new data; enables recording after first packet."""
        if not self.data_received:
            self.data_received = True
            self._update_controls()
        if self.recorder.recording:
            latest_time = times[-1]
            emg_vector = values[-1, :] if values.ndim > 1 else [values[-1]]
            self.recorder.record_data_point(latest_time, emg_vector)
    
    def handle_automatic_marker(self, timestamp):
        """Handle automatic events from the graph widget"""
        if self.recorder.recording:
            print(f"[MainWindow] Received automatic marker signal at timestamp: {timestamp:.3}s")
            # mark event with exact timestamp from the graph
            self.recorder.mark_event_with_timestamp(timestamp)
        else:
            print(f"[MainWindow] Automatic marker at timestamp: {timestamp:.3}s but recorder not recording")

    def update_countdown(self, countdown_text):
        """Update the countdown label in the top toolbar."""
        self.countdown_label.setText(countdown_text)
    
    def show_marker_interval_dialog(self):
        """Show dialog to set marker interval and enable automatic markers."""
        if not hasattr(self, 'graph') or not self.graph:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please initialize a graph first.")
            return False
        
        # Ask for interval - force the dialog to be modal and visible
        try:
            dialog = QtWidgets.QInputDialog(self)
            dialog.setWindowTitle("Set Marker Interval")
            dialog.setLabelText("Enter interval between markers (seconds):")
            dialog.setDoubleValue(getattr(self.graph, 'spike_interval', 5.0))
            dialog.setDoubleMinimum(0.1)
            dialog.setDoubleMaximum(60.0)
            dialog.setDoubleDecimals(1)
            dialog.setModal(True)
            
            result = dialog.exec_()
            
            if result == QtWidgets.QDialog.Accepted:
                interval = dialog.doubleValue()
                ok = True
            else:
                interval = 0
                ok = False
                
        except Exception as e:
            self.chat.log_event(f"Dialog error: {e}")
            return False
        
        if ok:
            self.graph.set_spike_interval(interval)
            self.graph.set_auto_spike_enabled(True)
            self.btn_spike_mode.setText("Disable Auto Markers")
            self.chat.log_event(f"Automatic markers enabled (every {interval}s)")
            return True
        else:
            # User cancelled
            return False

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
            # Enable auto spikes - use the shared dialog method
            if self.show_marker_interval_dialog():
                # Dialog was successful, state already updated in show_marker_interval_dialog
                pass
            else:
                # User cancelled, don't change state
                return

    def keyPressEvent(self, event):
        """Manual markers disabled, only automatic markers are active"""
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
        # Stop auto analysis timers
        if self.auto_analysis_enabled:
            self.stop_auto_analysis_timer()
        
        # Stop background processes
        self.stop_background_processes()
        self.output_timer.stop()
        
        if hasattr(self, 'graph') and self.graph.thread.isRunning():
            self.graph.thread.stop()
        if hasattr(self, 'live'):
            self.live.close()
        self.recorder.close()
        super().closeEvent(event)

    def start_background_process(self, script_path, process_name, output_queue):
        """Start a background process and capture its output."""
        def read_output(process, queue, name):
            """Read output from process and put it in queue."""
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.strip()
                        if line:
                            queue.put(f"[{name}] {line}")
                            print(f"[{name}] {line}")  # Also print to main terminal
                process.stdout.close()
            except Exception as e:
                queue.put(f"[{name}] Error reading output: {e}")
                print(f"[{name}] Error reading output: {e}")
        
        try:
            # Try to find and use the current Python environment
            # First, try to use the same Python interpreter that's running this script
            python_executable = sys.executable
            
            # If we're in a virtual environment, use it directly
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                # We're in a virtual environment, use the current Python
                command = f'"{python_executable}" "{script_path}"'
            else:
                # Not in a virtual environment, try to find a common venv path
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                possible_venv_paths = [
                    os.path.join(base_dir, "venv", "bin", "activate"),
                    os.path.join(base_dir, "env", "bin", "activate"),
                    os.path.join(base_dir, ".venv", "bin", "activate"),
                    "/home/phantasiai/Prototype/prot/bin/activate"  # Keep original as fallback
                ]
                
                venv_path = None
                for path in possible_venv_paths:
                    if os.path.exists(path):
                        venv_path = path
                        break
                
                if venv_path:
                    command = f'source "{venv_path}" && python3 "{script_path}"'
                    output_queue.put(f"[{process_name}] Using virtual environment: {venv_path}")
                else:
                    # No virtual environment found, use system Python
                    command = f'python3 "{script_path}"'
                    output_queue.put(f"[{process_name}] Using system Python")
            
            output_queue.put(f"[{process_name}] Executing command: {command}")
            
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start thread to read output
            output_thread = threading.Thread(
                target=read_output,
                args=(process, output_queue, process_name),
                daemon=True
            )
            output_thread.start()
            
            return process
        except Exception as e:
            output_queue.put(f"[{process_name}] Failed to start: {e}")
            print(f"[{process_name}] Failed to start: {e}")
            return None

    def stop_background_processes(self):
        """Stop all background processes."""
        if self.ai_process:
            try:
                self.ai_process.terminate()
                self.ai_process.wait(timeout=5)
            except:
                self.ai_process.kill()
            self.ai_process = None
            
        if self.stim_process:
            try:
                self.stim_process.terminate()
                self.stim_process.wait(timeout=5)
            except:
                self.stim_process.kill()
            self.stim_process = None

    def process_background_output(self):
        """Process output from background processes and display in chat."""
        # Process AI output
        while not self.ai_output_queue.empty():
            try:
                message = self.ai_output_queue.get_nowait()
                self.chat.log_event(message)
            except queue.Empty:
                break
                
        # Process STIM output
        while not self.stim_output_queue.empty():
            try:
                message = self.stim_output_queue.get_nowait()
                self.chat.log_event(message)
            except queue.Empty:
                break

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
        
        self.btn_sensor_type.setEnabled(self.data_type == "live")
        
        # Auto analysis button is enabled only when recording (if present)
        if hasattr(self, "btn_auto_analysis"):
            self.btn_auto_analysis.setEnabled(not dummy and self.recorder.recording)


        if hasattr(self.graph, 'btn_pause'):
            self.graph.btn_pause.setEnabled(not dummy)
        if hasattr(self.graph, 'btn_zoom_in'):
            self.graph.btn_zoom_in.setEnabled(not dummy)
        if hasattr(self.graph, 'btn_zoom_out'):
            self.graph.btn_zoom_out.setEnabled(not dummy)
        if hasattr(self.graph, 'btn_reset_zoom'):
            self.graph.btn_reset_zoom.setEnabled(not dummy)
