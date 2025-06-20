import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from graph_widget import GraphWidget
from chat_widget import ChatWidget
from emg_features_extractor import EMGFeatureExtractor
from startup_dialog import StartupDialog
from live_mode import run_live_mode

# Custom application-wide StyleSheet
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
    def __init__(self, mode, mode_label=""):
        super().__init__()
        self.mode = mode  
        title = "PhantasiAi"
        if mode_label:
            title += f" – {mode_label}"
        self.setWindowTitle(title)
        self._build_ui()

        self.setMinimumSize(1280, 720)

        if not self.charger_fichier_npz():
            QtWidgets.QMessageBox.critical(self, "Error", "No .npz file selected. Closing.")
            sys.exit(0)

    def _build_ui(self):
        conteneur = QtWidgets.QWidget()
        disposition_principale = QtWidgets.QVBoxLayout(conteneur)
        disposition_principale.setContentsMargins(0, 0, 0, 0)
        disposition_principale.setSpacing(0)

        # Toolbar buttons
        self.btn_changer         = QtWidgets.QPushButton("Open")
        self.btn_sauvegarder     = QtWidgets.QPushButton("Save")
        self.btn_canaux          = QtWidgets.QPushButton("Channels")
        self.btn_caracteristiques = QtWidgets.QPushButton("EMG Data")

        btn_size = QtCore.QSize(160, 60)
        font    = QtGui.QFont("Segoe UI", 7)
        for btn in (self.btn_changer, self.btn_sauvegarder, self.btn_canaux, self.btn_caracteristiques):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            btn.setStyleSheet("QPushButton { font-size: 7pt; padding: 7px; }")

        barre_outils = QtWidgets.QHBoxLayout()
        barre_outils.setContentsMargins(12, 12, 12, 6)
        barre_outils.setSpacing(8)
        barre_outils.addWidget(self.btn_changer)
        barre_outils.addWidget(self.btn_sauvegarder)
        barre_outils.addWidget(self.btn_canaux)
        barre_outils.addWidget(self.btn_caracteristiques)
        barre_outils.addStretch()
        disposition_principale.addLayout(barre_outils)

        # Body: graph + chat placeholder
        self.disposition_corps = QtWidgets.QHBoxLayout()
        self.disposition_corps.setContentsMargins(12, 6, 12, 12)
        self.disposition_corps.setSpacing(10)

        self.chat = ChatWidget(mode=self.mode)
        self.disposition_corps.addWidget(self.chat)

        disposition_principale.addLayout(self.disposition_corps)
        self.setCentralWidget(conteneur)

        # Connect toolbar signals
        self.btn_changer.clicked.connect(self.au_changement)
        self.btn_sauvegarder.clicked.connect(self.sauvegarder_logs)
        self.btn_canaux.clicked.connect(self.afficher_dialogue_canaux)
        self.btn_caracteristiques.clicked.connect(self.enregistrer_caracteristiques)

    def sauvegarder_logs(self):
        chemin, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Logs", "", "Text File (*.txt)"
        )
        if chemin:
            try:
                with open(chemin, 'w') as f:
                    f.write("\n".join(self.chat.get_logs()))
                self.chat.log_event(f"Logs saved to {os.path.basename(chemin)}")
                QtWidgets.QMessageBox.information(
                    self, "Save Successful",
                    f"Logs saved to {os.path.basename(chemin)}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def construire_corps(self):
        for i in reversed(range(self.disposition_corps.count())):
            w = self.disposition_corps.itemAt(i).widget()
            if isinstance(w, GraphWidget):
                w.thread.stop()
                w.setParent(None)

        # Prepare data
        mat = self.donnees_completes[:, self.canaux_selectionnes]
        self.nb_canaux = len(self.canaux_selectionnes)
        self.indice_donnees = 0

        def lire_vecteur():
            vec = mat[self.indice_donnees]
            self.indice_donnees = (self.indice_donnees + 1) % len(mat)
            return vec.tolist()

        etiquettes = [f"Channel {c+1}" for c in self.canaux_selectionnes]

        # Instantiate GraphWidget
        self.graphique = GraphWidget(
            read_func=lire_vecteur,
            num_channels=self.nb_canaux,
            channel_labels=etiquettes,
            title="EMG Channels",
            sample_rate=self.frequence_echantillonnage,
            buffer_seconds=self.secondes_tampon
        )
        self.graphique.misEnPause.connect(lambda: self.chat.log_event("Data stream paused"))
        self.graphique.repris.connect(lambda: self.chat.log_event("Data stream resumed"))

        for i in reversed(range(self.disposition_corps.count())):
            self.disposition_corps.takeAt(i)

        if self.mode == "pro":
            graph_stretch, chat_stretch = 4, 1
        else:
            graph_stretch, chat_stretch = 3, 2

        self.disposition_corps.addWidget(self.graphique, graph_stretch)
        self.disposition_corps.addWidget(self.chat, chat_stretch)

        liste = ', '.join(str(c+1) for c in self.canaux_selectionnes)
        self.chat.log_event(f"Graph initialized for channels: {liste}")

    def charger_fichier_npz(self):
        chemin, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a .npz file", "", "NumPy Archive (*.npz)"
        )
        if not chemin:
            return False
        try:
            with np.load(chemin) as ar:
                cle = 'emg' if 'emg' in ar.files else ar.files[0]
                tableau = ar[cle]
            if tableau.ndim == 1:
                tableau = tableau.reshape(-1, 1)
            self.donnees_completes = tableau
            self.frequence_echantillonnage = 220
            self.secondes_tampon = 2
            self.canaux_selectionnes = [0]
            self.chat.set_file(chemin)
            self.chat.log_event(f"Opened file « {os.path.basename(chemin)} »")
            self.construire_corps()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return False

    def au_changement(self):
        if self.charger_fichier_npz():
            self.chat.log_event("Data source changed")

    def afficher_dialogue_canaux(self):
        dialogue = QtWidgets.QDialog(self)
        dialogue.setWindowTitle("Select Channels")
        disposition = QtWidgets.QVBoxLayout(dialogue)

        tout = QtWidgets.QCheckBox("Select/Deselect All")
        tout.setChecked(len(self.canaux_selectionnes) == self.donnees_completes.shape[1])
        disposition.addWidget(tout)

        grille = QtWidgets.QGridLayout()
        self.cases = []
        nb = self.donnees_completes.shape[1]
        for idx in range(nb):
            cb = QtWidgets.QCheckBox(f"Canal {idx+1}")
            cb.setChecked(idx in self.canaux_selectionnes)
            self.cases.append(cb)
            grille.addWidget(cb, idx//4, idx%4)
        disposition.addLayout(grille)

        def toggle_all(state):
            for cb in self.cases:
                cb.blockSignals(True)
                cb.setChecked(bool(state))
                cb.blockSignals(False)

        def update_toggle():
            all_checked = all(cb.isChecked() for cb in self.cases)
            tout.blockSignals(True)
            tout.setChecked(all_checked)
            tout.blockSignals(False)

        tout.stateChanged.connect(toggle_all)
        for cb in self.cases:
            cb.stateChanged.connect(update_toggle)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.clicked.connect(dialogue.accept)
        btn_cancel.clicked.connect(dialogue.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        disposition.addLayout(btn_layout)

        if dialogue.exec_() == QtWidgets.QDialog.Accepted:
            sel = [i for i, cb in enumerate(self.cases) if cb.isChecked()]
            if sel:
                self.canaux_selectionnes = sel
                self.chat.log_event(f"Channels selected: {', '.join(str(i+1) for i in sel)}")
                self.construire_corps()

    def enregistrer_caracteristiques(self):
        mat = self.donnees_completes[:, self.canaux_selectionnes]
        ext = EMGFeatureExtractor(emg_array=mat)
        feats = ext.features_dict
        for idx, ch in enumerate(self.canaux_selectionnes):
            mav = feats['mav'][idx]
            rms = feats['rms'][idx]
            ssc = feats['ssc'][idx]
            var = feats['var'][idx]
            self.chat.log_event(
                f"Channel {ch+1} Features — MAV: {mav:.6f}, RMS: {rms:.6f}, SSC: {ssc}, VAR: {var:.6e}"
            )

    def closeEvent(self, event):
        if hasattr(self, 'graphique'):
            self.graphique.thread.stop()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    startup = StartupDialog()
    if startup.exec_() == QtWidgets.QDialog.Accepted:
        mode, data_type = startup.get_selections()
        mode_label = f"{mode.capitalize()} Mode"
        if data_type == "live":
            run_live_mode()
            return
        else:
            fen = MainWindow(mode=mode, mode_label=mode_label)
            fen.show()
            sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
