from PyQt5 import QtWidgets, QtCore, QtGui

class StartupDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhantasiAI - Configuration de démarrage")
        # Increased size for better readability
        self.setFixedSize(550, 380)
        self.setModal(True)

        # Set a larger default font
        default_font = QtGui.QFont("Segoe UI", 12)
        self.setFont(default_font)

        # Variables to store selections
        self.selected_mode = None
        self.selected_type = None

        self.setup_ui()

    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QtWidgets.QLabel("Configuration de PhantasiAI")
        title_font = QtGui.QFont(self.font().family(), 18, QtGui.QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        # Mode category
        mode_group = QtWidgets.QGroupBox("Mode")
        mode_group.setFont(QtGui.QFont(self.font().family(), 14, QtGui.QFont.Bold))
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        mode_layout.setSpacing(20)

        self.mode_group = QtWidgets.QButtonGroup()
        self.pro_mode_btn = QtWidgets.QRadioButton("Mode Pro")
        self.normal_mode_btn = QtWidgets.QRadioButton("Mode Normal")
        for btn in (self.pro_mode_btn, self.normal_mode_btn):
            btn.setFont(QtGui.QFont(self.font().family(), 13))
        self.mode_group.addButton(self.pro_mode_btn, 0)
        self.mode_group.addButton(self.normal_mode_btn, 1)
        mode_layout.addWidget(self.pro_mode_btn)
        mode_layout.addWidget(self.normal_mode_btn)
        main_layout.addWidget(mode_group)

        # Type category
        type_group = QtWidgets.QGroupBox("Type")
        type_group.setFont(QtGui.QFont(self.font().family(), 14, QtGui.QFont.Bold))
        type_layout = QtWidgets.QHBoxLayout(type_group)
        type_layout.setSpacing(20)

        self.type_group = QtWidgets.QButtonGroup()
        self.db_mode_btn = QtWidgets.QRadioButton("Base de données")
        self.live_mode_btn = QtWidgets.QRadioButton("Mode Live")
        for btn in (self.db_mode_btn, self.live_mode_btn):
            btn.setFont(QtGui.QFont(self.font().family(), 13))
        self.type_group.addButton(self.db_mode_btn, 0)
        self.type_group.addButton(self.live_mode_btn, 1)
        type_layout.addWidget(self.db_mode_btn)
        type_layout.addWidget(self.live_mode_btn)
        main_layout.addWidget(type_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QtWidgets.QPushButton("Annuler")
        self.cancel_btn.setFixedSize(120, 40)
        self.cancel_btn.setFont(QtGui.QFont(self.font().family(), 12))
        self.cancel_btn.clicked.connect(self.reject)

        self.next_btn = QtWidgets.QPushButton("Suivant")
        self.next_btn.setFixedSize(120, 40)
        self.next_btn.setFont(QtGui.QFont(self.font().family(), 12))
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.on_next_clicked)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.next_btn)
        main_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Enable Next only when both are selected
        self.mode_group.buttonClicked.connect(self.check_selections)
        self.type_group.buttonClicked.connect(self.check_selections)

    def check_selections(self):
        mode_selected = self.mode_group.checkedButton() is not None
        type_selected = self.type_group.checkedButton() is not None
        self.next_btn.setEnabled(mode_selected and type_selected)

    def on_next_clicked(self):
        mode_id = self.mode_group.checkedId()
        type_id = self.type_group.checkedId()
        self.selected_mode = "pro" if mode_id == 0 else "normal"
        self.selected_type = "database" if type_id == 0 else "live"
        self.accept()

    def get_selections(self):
        return self.selected_mode, self.selected_type
