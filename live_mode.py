import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import serial
from pyfirmata import Arduino, util
from pylsl import StreamInfo, StreamOutlet
import time
import threading
from collections import deque

board = None
analog_inputs = []
PORT_ARDUINO = "/dev/ttyUSB0"

def try_connect_arduino(port=PORT_ARDUINO):
    global board, analog_inputs
    try:
        board = Arduino(port)
        it = util.Iterator(board)
        it.start()
        analog_pins = ['a:0:i', 'a:1:i', 'a:2:i', 'a:3:i', 'a:4:i', 'a:5:i']
        analog_inputs = [board.get_pin(pin) for pin in analog_pins]
        return True
    except serial.SerialException:
        return False

def is_sensor_connected(pin, threshold=50):
    values = []
    for _ in range(10):
        sensor_value = pin.read()
        if sensor_value is not None:
            values.append(sensor_value * 1023)
    return max(values) > threshold

class DataThread(QtCore.QObject):
    dataUpdated = QtCore.pyqtSignal(list, list)

    def __init__(self, analog_input=None, seconds_to_display=1):
        super().__init__()
        self.analog_input = analog_input
        self.timestamp = 0
        self.buffer_emg = deque(maxlen=seconds_to_display * 220)
        self.timestamps_buffer = deque(maxlen=seconds_to_display * 220)
        self.running = True

        if analog_input is not None:
            self.info = StreamInfo('EMG', 'EMG', 1, 220, 'float32', 'myuid34234')
            self.outlet = StreamOutlet(self.info)

    def read_emg_data(self):
        sensor_value = self.analog_input.read()
        if sensor_value is not None:
            return int(sensor_value * 1023)
        return 0

    def data_generation_thread(self):
        while self.running:
            start = time.perf_counter()
            emg_value = self.read_emg_data()
            self.outlet.push_sample([emg_value])
            timestamp = self.timestamp / 220
            self.buffer_emg.append(emg_value)
            self.timestamps_buffer.append(timestamp)
            self.timestamp += 1
            self.dataUpdated.emit(list(self.buffer_emg), list(self.timestamps_buffer))
            end = time.perf_counter()
            time.sleep(max(1/220 - (end - start), 0))

    def stop(self):
        self.running = False

class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhantasiAI - Mode Live")
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(self.win)
        self.plots = []
        self.curves = []
        self.data_threads = []
        self.selected_analog_inputs = []

        self.button_layout = QtWidgets.QHBoxLayout()
        self.change_channels_button = QtWidgets.QPushButton("Modifier les canaux")
        self.change_channels_button.clicked.connect(self.change_selected_channels)
        self.button_layout.addWidget(self.change_channels_button)

        self.central_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.addWidget(self.win)
        self.main_layout.addLayout(self.button_layout)
        self.setCentralWidget(self.central_widget)

    def prompt_user_for_channel(self):
        channels_with_sensors = [i for i in range(len(analog_inputs)) if is_sensor_connected(analog_inputs[i])]
        if not channels_with_sensors:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucun capteur détecté.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Sélectionner les canaux")
        layout = QtWidgets.QVBoxLayout()
        checkboxes = []
        for i in channels_with_sensors:
            checkbox = QtWidgets.QCheckBox(f"Canal {i}")
            checkbox.setChecked(False)
            checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: self.handle_channel_selection(checkboxes, dialog))
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_channel_selection(self, checkboxes, dialog):
        selected_channels = [i for i, checkbox in enumerate(checkboxes) if checkbox.isChecked()]
        self.selected_analog_inputs = [analog_inputs[i] for i in selected_channels]

        for thread in self.data_threads:
            thread.stop()
        self.data_threads.clear()

        for plot in self.plots:
            self.win.removeItem(plot)
        self.plots.clear()
        self.curves.clear()

        for i, pin in enumerate(self.selected_analog_inputs):
            plot = self.win.addPlot(row=len(self.plots), col=0, title=f"EMG Signal - Canal {selected_channels[i]}")
            curve = plot.plot(pen='y')
            self.curves.append(curve)
            self.plots.append(plot)

            thread = DataThread(analog_input=pin)
            thread.dataUpdated.connect(lambda buffer_emg, timestamps_buffer, curve=curve: self.update_plot(buffer_emg, timestamps_buffer, curve))
            threading.Thread(target=thread.data_generation_thread, daemon=True).start()
            self.data_threads.append(thread)

        dialog.accept()

    def update_plot(self, buffer_emg, timestamps_buffer, curve):
        curve.setData(timestamps_buffer, buffer_emg)
        curve.getViewBox().setRange(xRange=[timestamps_buffer[0], timestamps_buffer[-1]], padding=0)

    def change_selected_channels(self):
        for thread in self.data_threads:
            thread.stop()
        self.data_threads.clear()

        for plot in self.plots:
            self.win.removeItem(plot)
        self.plots.clear()
        self.curves.clear()

        self.prompt_user_for_channel()

def run_live_mode():
    if not try_connect_arduino():
        QtWidgets.QMessageBox.critical(None, "Erreur", "Connexion à l'Arduino échouée.")
        return

    app = QtWidgets.QApplication([])
    window = LiveWindow()
    window.prompt_user_for_channel()
    window.show()
    app.exec()

    if board:
        board.exit()
