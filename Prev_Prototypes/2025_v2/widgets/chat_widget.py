import os
import time
import datetime
from PyQt5 import QtWidgets, QtCore, QtGui

class ChatWidget(QtWidgets.QFrame):
    """
    A custom PyQt5 widget that simulates a basic chatbot interface.
    Includes a header, file/date info, a message log area, and a message input field.
    """

    def __init__(self, mode="normal"):
        super().__init__()

        # Set frame styling
        self.setStyleSheet("""
            background:#FFFFFF;
            color:#000000;
            border:1px solid #CCC;
            border-radius:8px;
        """)
        self.logs = []  # Store log history

        # Adjust width based on mode
        width = 20 if mode == "pro" else 140 #220 #600
        self.setFixedWidth(width)

        # Main vertical layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Title label
        lbl = QtWidgets.QLabel("Chatbot")
        lbl.setFont(QtGui.QFont("", 12, QtGui.QFont.Bold))
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl)

        # File name display
        self.file_label = QtWidgets.QLabel("File : (none selected)")
        self.file_label.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.file_label)

        # File date display
        self.date_label = QtWidgets.QLabel("Date : (none)")
        self.date_label.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.date_label)

        # Log area: read-only text area showing messages
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFont("", 9))
        layout.addWidget(self.log, 1)  # Stretch factor = 1

        # Message input field
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Type your message here...")
        self.input.returnPressed.connect(self.send_message)
        self.input.setStyleSheet("padding-right: 30px; padding: 4px;")  # Leave space for the button

        # Create send button (arrow icon inside the input field)
        self.send_icon_button = QtWidgets.QToolButton(self.input)
        self.send_icon_button.setText("➤")
        self.send_icon_button.setStyleSheet("color: #A523DC; background: transparent; border: none; font-size: 16px;")
        self.send_icon_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.send_icon_button.clicked.connect(self.send_message)

        # Dynamically position the send button inside the input field
        def resizeEvent(old_resize):
            def new_resize(event):
                old_resize(event)
                button_size = self.send_icon_button.sizeHint()
                self.send_icon_button.move(
                    self.input.width() - button_size.width() - 5,
                    (self.input.height() - button_size.height()) // 2
                )
            return new_resize
        self.input.resizeEvent = resizeEvent(self.input.resizeEvent)

        # Layout for input field (optional placeholder for extensions)
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)
        input_layout.addWidget(self.input)

        # Final input field styling and placement
        self.input.setFixedHeight(28)
        self.input.setFont(QtGui.QFont("", 9))
        layout.addLayout(input_layout)
        layout.addWidget(self.input)

    @QtCore.pyqtSlot(str)
    def set_file(self, path):
        """
        Set the file info labels based on a selected file path.
        Displays file name and creation date.
        """
        nom = os.path.basename(path)
        self.file_label.setText(f"File : {nom}")

        ctime = time.localtime(os.path.getctime(path))
        datestr = time.strftime("%Y-%m-%d %H:%M:%S", ctime)
        self.date_label.setText(f"Date : {datestr}")

    def send_message(self):
        """
        Called when the user presses Enter or clicks the send button.
        Displays user input and a dummy response.
        """
        message = self.input.text().strip()
        if message:
            self.log_event(f"You: {message}")
            self.input.clear()
            self.log_event(f"Message received: '{message}'")

    def get_logs(self):
        """
        Return the list of all logged messages (timestamps included).
        """
        return self.logs

    @QtCore.pyqtSlot(str)
    def log_event(self, message):
        """
        Add a new message to the log with a timestamp.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        self.log.append(entry)
    
    def set_mode(self, mode):
        """
        Adjust visibility based on data mode.
        In 'live' mode, hide file and date labels.
        """
        if mode == "live":
            self.file_label.hide()
            self.date_label.hide()
        else:
            self.file_label.show()
            self.date_label.show()

