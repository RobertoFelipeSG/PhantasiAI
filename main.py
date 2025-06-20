import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from graph_widget import GraphWidget
from chat_widget import ChatWidget
from emg_features_extractor import EMGFeatureExtractor

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

class FenetrePrincipale(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhantasiAi")
        self.setFixedSize(1280, 720)

        # Main container and layout
        conteneur = QtWidgets.QWidget()
        disposition_principale = QtWidgets.QVBoxLayout(conteneur)
        disposition_principale.setContentsMargins(0, 0, 0, 0)
        disposition_principale.setSpacing(0)

        # Toolbar buttons
        self.btn_changer = QtWidgets.QPushButton("Ouvrir")
        self.btn_sauvegarder = QtWidgets.QPushButton("Enregistrer")
        self.btn_canaux = QtWidgets.QPushButton("Canaux")
        self.btn_caracteristiques = QtWidgets.QPushButton("Données EMG")
        
        btn_size = QtCore.QSize(160, 60)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(7)  
        
        for btn in (self.btn_changer, self.btn_sauvegarder, self.btn_canaux, self.btn_caracteristiques):
            btn.setFixedSize(btn_size)
            btn.setFont(font)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 7pt;
                    padding: 7px;
                }
            """)

        # Toolbar layout
        barre_outils = QtWidgets.QHBoxLayout()
        barre_outils.setContentsMargins(12, 12, 12, 6)
        barre_outils.setSpacing(8)
        barre_outils.addWidget(self.btn_changer)
        barre_outils.addWidget(self.btn_sauvegarder)
        barre_outils.addWidget(self.btn_canaux)
        barre_outils.addWidget(self.btn_caracteristiques)
        barre_outils.addStretch()
        disposition_principale.addLayout(barre_outils)

        # Body: graph + chat
        self.disposition_corps = QtWidgets.QHBoxLayout()
        self.disposition_corps.setContentsMargins(12, 6, 12, 12)
        self.disposition_corps.setSpacing(10)
        self.chat = ChatWidget()
        self.disposition_corps.addWidget(self.chat, 1)
        disposition_principale.addLayout(self.disposition_corps)

        self.setCentralWidget(conteneur)

        # Signals
        self.btn_changer.clicked.connect(self.au_changement)
        self.btn_sauvegarder.clicked.connect(self.sauvegarder_logs)
        self.btn_canaux.clicked.connect(self.afficher_dialogue_canaux)
        self.btn_caracteristiques.clicked.connect(self.enregistrer_caracteristiques)

        # Load data
        if not self.charger_fichier_npz():
            QtWidgets.QMessageBox.critical(self, "Erreur", "Aucun fichier .npz sélectionné. Fermeture.")
            sys.exit(0)

    def sauvegarder_logs(self):
        chemin, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer les logs", "", "Fichier texte (*.txt)"
        )
        if chemin:
            try:
                with open(chemin, 'w') as f:
                    f.write("\n".join(self.chat.get_logs()))
                self.chat.log_event(f"Logs enregistrées dans {os.path.basename(chemin)}")
                QtWidgets.QMessageBox.information(
                    self, "Enregistrement réussi",
                    f"Logs enregistrées dans {os.path.basename(chemin)}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Erreur d'enregistrement", str(e))

    def construire_corps(self):
        if hasattr(self, 'graphique'):
            self.graphique.thread.stop()
            self.graphique.setParent(None)

        mat = self.donnees_completes[:, self.canaux_selectionnes]
        self.nb_canaux = len(self.canaux_selectionnes)
        self.indice_donnees = 0

        def lire_vecteur():
            vec = mat[self.indice_donnees]
            self.indice_donnees = (self.indice_donnees + 1) % len(mat)
            return vec.tolist()

        etiquettes = [f"Canal {c+1}" for c in self.canaux_selectionnes]

        self.graphique = GraphWidget(
            read_func=lire_vecteur,
            num_channels=self.nb_canaux,
            channel_labels=etiquettes,
            title="Canaux EMG",
            sample_rate=self.frequence_echantillonnage,
            buffer_seconds=self.secondes_tampon
        )
        self.graphique.misEnPause.connect(
            lambda: self.chat.log_event("Flux de données en pause")
        )
        self.graphique.repris.connect(
            lambda: self.chat.log_event("Reprise du flux de données")
        )
        self.disposition_corps.insertWidget(0, self.graphique, 1)

        liste_canaux = ', '.join(str(c+1) for c in self.canaux_selectionnes)
        self.chat.log_event(f"Graphique initialisé pour canaux : {liste_canaux}")

    def charger_fichier_npz(self):
        chemin, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Sélectionnez un fichier .npz", "", "Archive NumPy (*.npz)"
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
            self.chat.log_event(f"Ouverture du fichier « {os.path.basename(chemin)} »")
            self.construire_corps()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))
            return False

    def au_changement(self):
        if self.charger_fichier_npz():
            self.chat.log_event("Source de données changée")

    def afficher_dialogue_canaux(self):
        dialogue = QtWidgets.QDialog(self)
        dialogue.setWindowTitle("Sélection des canaux")
        disposition = QtWidgets.QVBoxLayout(dialogue)

        tout_selectionner = QtWidgets.QCheckBox("Sélectionner/Désélectionner tout")
        tout_selectionner.setChecked(
            len(self.canaux_selectionnes) == self.donnees_completes.shape[1]
        )
        disposition.addWidget(tout_selectionner)

        grille = QtWidgets.QGridLayout()
        self.cases_a_cocher = []
        nb_canaux = self.donnees_completes.shape[1]
        for idx in range(nb_canaux):
            cb = QtWidgets.QCheckBox(f"Canal {idx+1}")
            cb.setChecked(idx in self.canaux_selectionnes)
            self.cases_a_cocher.append(cb)
            grille.addWidget(cb, idx//4, idx%4)
        disposition.addLayout(grille)

        def au_tout_selectionner(etat):
            for cb in self.cases_a_cocher:
                cb.blockSignals(True)
                cb.setChecked(bool(etat))
                cb.blockSignals(False)

        def au_changement_case():
            tous_coches = all(cb.isChecked() for cb in self.cases_a_cocher)
            tout_selectionner.blockSignals(True)
            tout_selectionner.setChecked(tous_coches)
            tout_selectionner.blockSignals(False)

        tout_selectionner.stateChanged.connect(au_tout_selectionner)
        for cb in self.cases_a_cocher:
            cb.stateChanged.connect(au_changement_case)

        boutons = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("OK")
        btn_annuler = QtWidgets.QPushButton("Annuler")
        btn_ok.clicked.connect(dialogue.accept)
        btn_annuler.clicked.connect(dialogue.reject)
        boutons.addStretch()
        boutons.addWidget(btn_ok)
        boutons.addWidget(btn_annuler)
        disposition.addLayout(boutons)

        if dialogue.exec_() == QtWidgets.QDialog.Accepted:
            sel = [i for i, cb in enumerate(self.cases_a_cocher) if cb.isChecked()]
            if sel:
                self.canaux_selectionnes = sel
                self.chat.log_event(
                    f"Canaux sélectionnés : {', '.join(str(i+1) for i in sel)}"
                )
                self.construire_corps()

    def enregistrer_caracteristiques(self):
        mat = self.donnees_completes[:, self.canaux_selectionnes]
        extracteur = EMGFeatureExtractor(emg_array=mat)
        caract = extracteur.features_dict
        for idx, ch in enumerate(self.canaux_selectionnes):
            mav = caract['mav'][idx]
            rms = caract['rms'][idx]
            ssc = caract['ssc'][idx]
            var = caract['var'][idx]
            self.chat.log_event(
                f"Caractéristiques Canal {ch+1} — MAV: {mav:.6f}, RMS: {rms:.6f}, SSC: {ssc}, VAR: {var:.6e}"
            )

    def closeEvent(self, evenement):
        if hasattr(self, 'graphique'):
            self.graphique.thread.stop()
        super().closeEvent(evenement)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    fenetre = FenetrePrincipale()
    fenetre.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
