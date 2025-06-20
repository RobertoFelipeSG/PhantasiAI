import sys
import os
from PyQt5 import QtWidgets
import numpy as np

from graph_widget import GraphWidget
from chat_widget import ChatWidget
from startup_dialog import StartupDialog
from live_mode import run_live_mode  


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode, data_type):
        super().__init__()
        self.mode = mode
        self.data_type = data_type

        self.setWindowTitle(f"PhantasiAi - {mode.capitalize()} Mode")
        self.setFixedSize(1280, 720)

        container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.btn_change = QtWidgets.QPushButton("Ouvrir")
        self.btn_save = QtWidgets.QPushButton("Enregistrer")
        self.btn_channels = QtWidgets.QPushButton("Channels")

        buttons = [self.btn_change, self.btn_save, self.btn_channels]
        for btn in buttons:
            btn.setFixedSize(120, 36)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(8, 8, 8, 4)
        top.setSpacing(6)
        top.addWidget(self.btn_change)
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_channels)
        top.addStretch()
        main_layout.addLayout(top)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(8, 4, 8, 8)
        body.setSpacing(8)

        self.chat = ChatWidget(mode)
        body.addWidget(self.chat, 1)

        main_layout.addLayout(body)
        self.setCentralWidget(container)

        self.btn_change.clicked.connect(self.on_change)
        self.btn_save.clicked.connect(lambda: self.chat.log_event("Enregistrement demandé"))
        self.btn_save.clicked.connect(self.save_logs)
        self.btn_channels.clicked.connect(lambda: self.chat.log_event("Gestion des canaux demandée"))
        self.btn_channels.clicked.connect(self.show_channels_dialog)

        self.chat.log_event(f"Application lancée en mode {mode} avec type {data_type}")

        if not self.load_npz_file():
            QtWidgets.QMessageBox.critical(self, "Erreur", "Aucun fichier .npz sélectionné. Fermeture.")
            sys.exit(0)

    def save_logs(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer les logs", "", "Fichier texte (*.txt)"
        )
        if path:
            try:
                with open(path, 'w') as file:
                    logs = self.chat.get_logs()
                    file.write('\n'.join(logs))
                self.chat.log_event(f"Logs enregistrées dans {os.path.basename(path)}")
                QtWidgets.QMessageBox.information(self, "Enregistrement réussi", f"Logs enregistrées dans {os.path.basename(path)}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Erreur d'enregistrement", str(e))

    def build_body(self):
        if hasattr(self, 'graph') and self.graph:
            self.graph.setParent(None)

        def read_array():
            v = self.data[self.data_idx]
            self.data_idx = (self.data_idx + 1) % len(self.data)
            return float(v)

        self.graph = GraphWidget(read_array, title="EMG Temps Réel")
        self.graph.misEnPause.connect(lambda: self.chat.log_event("Flux de données en pause"))
        self.graph.repris.connect(lambda: self.chat.log_event("Reprise du flux de données"))

        body = self.centralWidget().layout().itemAt(1)
        body.insertWidget(0, self.graph, 3)
        self.chat.log_event("Widget graphique initialisé")

    def load_npz_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Sélectionnez un fichier .npz", "", "Archive NumPy (*.npz)"
        )
        if not path:
            return False
        try:
            with np.load(path) as ar:
                if 'emg' in ar.files:
                    arr = ar['emg']
                    if arr.size:
                        self.data = arr.flatten()
                        self.data_idx = 0
                else:
                    for k in ar.files:
                        arr = ar[k]
                        if arr.size:
                            self.data = arr.flatten()
                            self.data_idx = 0
                            break
            self.chat.set_file(path)
            self.chat.log_event(f"Ouverture du fichier « {os.path.basename(path)} »")
            self.build_body()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))
            return False

    def on_change(self):
        if self.load_npz_file():
            self.graph.thread.buf.clear()
            self.graph.thread.tbuf.clear()
            self.chat.log_event("Source de données changée")

    def closeEvent(self, e):
        if hasattr(self, 'graph') and self.graph:
            self.graph.thread.stop()
        super().closeEvent(e)

    def show_channels_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Sélection des canaux")
        dialog.setFixedSize(400, 250)

        layout = QtWidgets.QVBoxLayout(dialog)
        button_layout = QtWidgets.QHBoxLayout()

        for i in range(1, 6):
            channel_btn = QtWidgets.QPushButton(f"Canal {i}")
            channel_btn.setFixedSize(70, 70)
            channel_btn.clicked.connect(lambda checked, ch=i: self.select_channel(ch))
            button_layout.addWidget(channel_btn)

        layout.addLayout(button_layout)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        dialog.exec_()

    def select_channel(self, channel):
        self.chat.log_event(f"Canal {channel} sélectionné")
        

def main():
    app = QtWidgets.QApplication(sys.argv)

    startup = StartupDialog()
    if startup.exec_() == QtWidgets.QDialog.Accepted:
        mode, data_type = startup.get_selections()

        if data_type == "live":
            run_live_mode()
            return
        else:
            w = MainWindow(mode, data_type)
            w.show()
            sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
