# main.py
import sys
from PyQt5 import QtWidgets, QtGui

from utils.scipy_patch import patch_scipy_welch
patch_scipy_welch()

from widgets.startup_widget import StartupDialog
from widgets.main_window import MainWindow

from utils.style import APP_STYLESHEET

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    dialog = StartupDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        mode, data_type = dialog.get_selections()
        mode_label = f"{mode.capitalize()} Mode"
        window = MainWindow(mode=mode, mode_label=mode_label, data_type=data_type)
        window.resize(1400, 800)
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
