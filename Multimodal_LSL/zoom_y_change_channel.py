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
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

board = None
analog_inputs = []

PORT_ARDUINO = "/dev/ttyUSB0"  # Ex: COM5 sous Windows



def try_connect_arduino(port='/dev/ttyUSB0'):
    print('essai1')
    global board, analog_inputs
    try:
        # Initialiser l'Arduino avec pyFirmata
        board = Arduino(port)  # Remplacez 'COMX' par le port série correct

        # Démarrer l'itérateur pour lire les entrées
        it = util.Iterator(board)
        it.start()

        # Configurez les broches analogiques en entrée
        analog_pins = ['a:0:i', 'a:1:i', 'a:2:i', 'a:3:i', 'a:4:i', 'a:5:i']
        analog_inputs = [board.get_pin(pin) for pin in analog_pins]
        return True

    except serial.SerialException:
        return False

is_arduino_connected = try_connect_arduino()

def is_sensor_connected(pin, threshold=50):
    """Checks if a sensor is connected to the selected pin."""
    values = []
    print('entree')
    for _ in range(10):  # Read multiple values to ensure sensor connection
        sensor_value = pin.read()
        if sensor_value is not None:
            values.append(sensor_value * 1023)  # Convert to Arduino range (0-1023)
    return max(values) > threshold  # Check if the values exceed the threshold

def load_npz_files(directory):
    data = []
    max_columns = 0
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            filepath = os.path.join(directory, filename)
            with np.load(filepath) as npzfile:
                for key in npzfile.files:
                    array = npzfile[key]
                    if array.ndim == 1:
                        array = array[:, np.newaxis]
                    if array.ndim == 2:
                        max_columns = max(max_columns, array.shape[1])
                    elif array.ndim == 3:
                        max_columns = max(max_columns, array.shape[1] * array.shape[2])
                    data.append(array)
    adjusted_data = []
    for array in data:
        if array.ndim == 1:
            array = array[:, np.newaxis]
        if array.ndim == 2:
            if array.shape[1] < max_columns:
                padding = np.zeros((array.shape[0], max_columns - array.shape[1]))
                array = np.hstack((array, padding))
        elif array.ndim == 3:
            array = array.reshape(array.shape[0], -1)
            if array.shape[1] < max_columns:
                padding = np.zeros((array.shape[0], max_columns - array.shape[1]))
                array = np.hstack((array, padding))
        adjusted_data.append(array)
    return np.concatenate(adjusted_data, axis=0)

class DataThread(QtCore.QObject):
    dataUpdated = QtCore.pyqtSignal(list, list)

    def __init__(self, analog_input=None, data=None, channel=None, seconds_to_display=1):
        super().__init__()
        self.analog_input = analog_input
        self.data = data
        self.channel = channel
        self.timestamp = 0
        self.buffer_emg = deque(maxlen=seconds_to_display * 220)
        self.timestamps_buffer = deque(maxlen=seconds_to_display * 220)
        self.running = True

        if analog_input is not None:
            self.info = StreamInfo('EMG', 'EMG', 1, 220, 'float32', 'myuid34234')
            self.outlet = StreamOutlet(self.info)

        with open("emg_data_2.txt", "w") as file:
            file.write("EMG, Timestamp\n")

    def read_emg_data(self):
        sensor_value = self.analog_input.read()
        if sensor_value is not None:
            sensor_value = int(sensor_value * 1023)
            return sensor_value
        return 0

    def log_data(self, timestamp, emg):
        with open("emg_data_2.1.txt", "a") as file:
            file.write(f"{emg},{timestamp}\n")

    def data_generation_thread(self):
        while self.running:
            start = time.perf_counter()
            if self.analog_input is not None:
                emg_value = self.read_emg_data()
                self.outlet.push_sample([emg_value])
            else:
                emg_value = self.data[self.timestamp // 1000, self.channel]
            
            #timestamp = time.time()  # timestamp réel en secondes (flottant)
            timestamp = self.timestamp / 220
            self.buffer_emg.append(emg_value)
            self.timestamps_buffer.append(timestamp)
            self.timestamp += 1
            


            self.log_data(timestamp, emg_value)

            self.dataUpdated.emit(list(self.buffer_emg), list(self.timestamps_buffer))

            end = time.perf_counter()
            time.sleep(max(1/220 - (end - start), 0))

    def stop(self):
        self.running = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time EMG Data")
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(self.win)
        self.plots = []
        self.curves = []
        self.data_threads = []
        self.selected_analog_inputs = []
        
        self.zoom_factor = 1.2  # Facteur de zoom
        self.y_min, self.y_max = 0, 1100  # Limites initiales de l'axe Y

        # Ajout d'un layout pour les boutons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.zoom_in_button = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom -")

        self.zoom_in_button.clicked.connect(self.zoom_in_y)
        self.zoom_out_button.clicked.connect(self.zoom_out_y)

        self.button_layout.addWidget(self.zoom_in_button)
        self.button_layout.addWidget(self.zoom_out_button)
        
        self.change_channels_button = QtWidgets.QPushButton("Modifier les canaux")
        self.change_channels_button.clicked.connect(self.change_selected_channels)
        self.button_layout.addWidget(self.change_channels_button)



        self.central_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.addWidget(self.win)
        self.main_layout.addLayout(self.button_layout)
        self.setCentralWidget(self.central_widget)
        


    def prompt_user_for_channel(self):
        """Ouvre une boîte de dialogue permettant de sélectionner ou de modifier les canaux affichés"""
        channels_with_sensors = [i for i in range(len(analog_inputs)) if is_sensor_connected(analog_inputs[i])]
        if not channels_with_sensors:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucun capteur détecté sur les canaux disponibles.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Sélectionner les canaux")

        layout = QtWidgets.QVBoxLayout()
        checkboxes = []
        for i in channels_with_sensors:
            checkbox = QtWidgets.QCheckBox(f"Canal {i}")
            checkbox.setChecked(i in [analog_inputs.index(pin) for pin in self.selected_analog_inputs])
            checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(lambda: self.handle_channel_selection(checkboxes, dialog))
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_channel_selection(self, checkboxes, dialog):
        """Met à jour les canaux affichés en fonction de la sélection utilisateur"""
        selected_channels = [i for i, checkbox in enumerate(checkboxes) if checkbox.isChecked()]
        self.selected_analog_inputs = [analog_inputs[i] for i in selected_channels]

        # Nettoyer les anciens graphes et threads
        for thread in self.data_threads:
            thread.stop()
        self.data_threads.clear()

        for plot in self.plots:
            self.win.removeItem(plot)
        self.plots.clear()
        self.curves.clear()

        # Ajouter les nouveaux graphes
        for i, pin in enumerate(self.selected_analog_inputs):
            plot = self.win.addPlot(row=len(self.plots), col=0, title=f"EMG Signal - Channel {selected_channels[i]}")
            curve = plot.plot(pen='y')
            self.curves.append(curve)
            self.plots.append(plot)

            thread = DataThread(analog_input=pin)
            thread.dataUpdated.connect(lambda buffer_emg, timestamps_buffer, curve=curve: self.update_plot(buffer_emg, timestamps_buffer, curve))
            threading.Thread(target=thread.data_generation_thread, daemon=True).start()
            self.data_threads.append(thread)

        dialog.accept()


    
    def zoom_in_y(self):
        self.y_min /= self.zoom_factor
        self.y_max /= self.zoom_factor
        self.update_y_range()

    def zoom_out_y(self):
        self.y_min *= self.zoom_factor
        self.y_max *= self.zoom_factor
        self.update_y_range()

    def update_y_range(self):
        for plot in self.plots:
            plot.setYRange(self.y_min, self.y_max, padding=0)
    
    def change_selected_channels(self):
        print("ok")
        # Supprimer les anciens plots
        for plot in self.plots:
            self.win.removeItem(plot)
        self.plots.clear()
        self.curves.clear()

        # Arrêter les anciens threads
        for thread in self.data_threads:
            thread.stop()
        self.data_threads.clear()

        # Relancer la sélection des canaux
        self.prompt_user_for_channel()



    @QtCore.pyqtSlot(list, list, object)
    def update_plot(self, buffer_emg, timestamps_buffer, curve):
        curve.setData(timestamps_buffer, buffer_emg)
        curve.getViewBox().setRange(xRange=[timestamps_buffer[0], timestamps_buffer[-1]], padding=0)

def main():
    import sys
    
    global data_thread

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()

    while True:
        mode, ok = QtWidgets.QInputDialog.getItem(None, "Select Mode", "Choose mode:", ["Real-time", "Database"], 0, False)
        if not ok:
            return

        if mode == "Real-time":
            print("ok32")
            if is_arduino_connected:
                print("ok3")
                connected_channels = [i for i in range(len(analog_inputs)) if is_sensor_connected(analog_inputs[i])]

                if not connected_channels:
                    QtWidgets.QMessageBox.warning(None, "No Sensors Detected", "No sensors are connected. Please connect sensors or switch to Database mode.")
                    continue

                print("Avant prompt_user_for_channel()")
                main_window.prompt_user_for_channel()
                print("Après prompt_user_for_channel()")

                main_window.show()
                sys.exit(app.exec())

        elif mode == "Database":
            db_path = '../data/dataset_v2_blocks/health/left/alex_kovalev_standart_elbow_left/preproc_angles/test'
            data = load_npz_files(db_path)

            num_channels = data.shape[1]
            channel, ok = QtWidgets.QInputDialog.getInt(None, "Select Channel", f"Choose a channel (0 to {num_channels-1}):", 0, 0, num_channels-1, 1)
            if not ok:
                return

            plot = main_window.win.addPlot(title=f"Channel {channel}")
            # ESSAIE D'AJOUTER CECI ICI :
            plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            curve = plot.plot(pen='y')

            buffer = deque(maxlen=1000 * 1000)
            timestamps_buffer = []

            def data_generation_thread(buffer=buffer, timestamps_buffer=timestamps_buffer, data=data, channel=channel):
                index = 0
                timestamp = 0.0

                while index < len(data):
                    chunk = data[index:index + 1000, channel]
                    buffer.extend(chunk)
                    timestamps_buffer.extend([(timestamp + i * 0.001) for i in range(len(chunk))])

                    timestamp += len(chunk) * 0.001
                    index += len(chunk)
                    #time.sleep(0.01)


            def update_plot(buffer=buffer, timestamps_buffer=timestamps_buffer, curve=curve, plot=plot):
                
                if len(buffer) and len(timestamps_buffer):
                    
                    # Use length of buffer to extract the last 50,000 elements if available
                    latest_data = np.array(list(buffer)[-100000:])
                    latest_timestamps = np.array(list(timestamps_buffer)[-100000:])
                   
                    if len(latest_data) == len(latest_timestamps):
                        curve.setData(latest_timestamps, latest_data)

                        plot.getAxis('bottom').setTickSpacing(5, 5)
                        plot.setRange(xRange=[latest_timestamps[0], latest_timestamps[-1]], padding=0)

                        for item in plot.items[:]:
                            if isinstance(item, pg.InfiniteLine) and item.value() < latest_timestamps[0]:
                                plot.removeItem(item)

                        marker_interval = 100
                        for t in range(int(latest_timestamps[0] // marker_interval) * marker_interval, int(latest_timestamps[-1]) + marker_interval, marker_interval):
                            if t >= latest_timestamps[0] and t <= latest_timestamps[-1]:
                                clock_marker = pg.InfiniteLine(t, angle=90, pen='r')
                                plot.addItem(clock_marker)

                # Update every x ms
                QtCore.QTimer.singleShot(1, lambda: update_plot(buffer, timestamps_buffer, curve, plot))


            threading.Thread(target=data_generation_thread, args=(buffer, timestamps_buffer, data, channel), daemon=True).start()
            update_plot(buffer, timestamps_buffer, curve, plot)
            

            main_window.show()
            sys.exit(app.exec())

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if is_arduino_connected:
            board.exit()


