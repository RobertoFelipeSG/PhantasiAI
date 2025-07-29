# main.py
import sys
from PyQt5 import QtWidgets, QtGui

from widgets.main_window import MainWindow
from utils.style import APP_STYLESHEET
from utils.scipy_patch import patch_scipy_welch

patch_scipy_welch()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)
    app.setWindowIcon(QtGui.QIcon("assets/favicon.ico"))
    window = MainWindow(mode="chat", mode_label="Live or Offline", data_type="live", file_path=None)
    window.resize(1400, 800)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
