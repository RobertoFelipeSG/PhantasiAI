from PyQt5 import QtWidgets, QtCore, QtGui

class StartupDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhantasiAI - Configuration de démarrage")
        self.setFixedSize(450, 300)
        self.setModal(True)
        
        # Variables to store selections
        self.selected_mode = None
        self.selected_type = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QtWidgets.QLabel("Configuration de PhantasiAI")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Mode category
        mode_group = QtWidgets.QGroupBox("Mode")
        mode_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        
        self.mode_group = QtWidgets.QButtonGroup()
        self.pro_mode_btn = QtWidgets.QRadioButton("Mode Pro")
        self.normal_mode_btn = QtWidgets.QRadioButton("Mode Normal")
        
        self.mode_group.addButton(self.pro_mode_btn, 0)
        self.mode_group.addButton(self.normal_mode_btn, 1)
        
        mode_layout.addWidget(self.pro_mode_btn)
        mode_layout.addWidget(self.normal_mode_btn)
        
        main_layout.addWidget(mode_group)
        
        # Type category
        type_group = QtWidgets.QGroupBox("Type")
        type_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        type_layout = QtWidgets.QHBoxLayout(type_group)
        
        self.type_group = QtWidgets.QButtonGroup()
        self.db_mode_btn = QtWidgets.QRadioButton("Base de données")
        self.live_mode_btn = QtWidgets.QRadioButton("Mode Live")
        
        self.type_group.addButton(self.db_mode_btn, 0)
        self.type_group.addButton(self.live_mode_btn, 1)
        
        type_layout.addWidget(self.db_mode_btn)
        type_layout.addWidget(self.live_mode_btn)
        
        main_layout.addWidget(type_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QtWidgets.QPushButton("Annuler")
        self.cancel_btn.setFixedSize(100, 35)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.next_btn = QtWidgets.QPushButton("Suivant")
        self.next_btn.setFixedSize(100, 35)
        self.next_btn.setEnabled(False)  # Disabled by default
        self.next_btn.clicked.connect(self.on_next_clicked)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.next_btn)
        
        main_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Connect signals to check if both selections are made
        self.mode_group.buttonClicked.connect(self.check_selections)
        self.type_group.buttonClicked.connect(self.check_selections)
        
    def check_selections(self):
        """Enable Next button only if both categories have a selection"""
        mode_selected = self.mode_group.checkedButton() is not None
        type_selected = self.type_group.checkedButton() is not None
        
        self.next_btn.setEnabled(mode_selected and type_selected)
        
    def on_next_clicked(self):
        # Get selections
        mode_id = self.mode_group.checkedId()
        type_id = self.type_group.checkedId()
        
        self.selected_mode = "pro" if mode_id == 0 else "normal"
        self.selected_type = "database" if type_id == 0 else "live"
        
        self.accept()
        
    def get_selections(self):
        """Returns the selected mode and type"""
        return self.selected_mode, self.selected_type