import os
import time
import datetime
from PyQt5 import QtWidgets, QtCore, QtGui

class ChatWidget(QtWidgets.QFrame):
    def __init__(self, mode="normal"):
        super().__init__()
        self.setStyleSheet("""
            background:#FFFFFF;
            color:#000000;
            border:1px solid #CCC;
            border-radius:8px;
        """)
        self.logs = []
        
        width = 220 if mode == "pro" else 600
        self.setFixedWidth(width)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6,6,6,6)
        layout.setSpacing(6)

        lbl = QtWidgets.QLabel("Chatbot")
        lbl.setFont(QtGui.QFont("", 12, QtGui.QFont.Bold))
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl)

        self.file_label = QtWidgets.QLabel("File : (none selected)")
        self.file_label.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.file_label)

        self.date_label = QtWidgets.QLabel("Date : (none)")
        self.date_label.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.date_label)

        # log area
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.log, 1)

        # saisie
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Type your message here...")
        self.input.returnPressed.connect(self.send_message)
        # Style the input to make room for the button
        self.input.setStyleSheet("padding-right: 30px; padding: 4px;")

        # Create a button with arrow icon
        self.send_icon_button = QtWidgets.QToolButton(self.input)
        self.send_icon_button.setText("➤")  
        self.send_icon_button.setStyleSheet("color: #A523DC; background: transparent; border: none; font-size: 16px;")
        self.send_icon_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.send_icon_button.clicked.connect(self.send_message)

        # Position the button inside the input field
        def resizeEvent(old_resize):
            def new_resize(event):
                old_resize(event)
                # Position button at right side of input, aligned vertically
                button_size = self.send_icon_button.sizeHint()
                self.send_icon_button.move(
                    self.input.width() - button_size.width() - 5,
                    (self.input.height() - button_size.height()) // 2
                )
            return new_resize

        self.input.resizeEvent = resizeEvent(self.input.resizeEvent)
        
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)
        input_layout.addWidget(self.input)
        
        layout.addLayout(input_layout)
        self.input.setStyleSheet("padding: 4px;")
        self.input.setFixedHeight(28)
        self.input.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.input)

    @QtCore.pyqtSlot(str)
    def set_file(self, path):
        nom = os.path.basename(path)
        self.file_label.setText(f"File : {nom}")
        ctime = time.localtime(os.path.getctime(path))
        datestr = time.strftime("%Y-%m-%d %H:%M:%S", ctime)
        self.date_label.setText(f"Date : {datestr}")


    def send_message(self):
        message = self.input.text().strip()
        if message:
            self.log_event(f"You: {message}")
            self.input.clear()
            self.log_event(f"Message received: '{message}'")

    def get_logs(self):
        return self.logs

    @QtCore.pyqtSlot(str)
    def log_event(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        self.log.append(entry)
