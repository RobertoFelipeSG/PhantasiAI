from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QRadioButton, QPushButton, QButtonGroup,
                             QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import platform
# Handle both direct execution and module imports
try:
    from ..offline_analysis.peak_classifier import PeakClassifier 
    from ..offline_analysis import BatchAnalyzer
except ImportError:
    # For direct execution, add parent directories to path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from Previous_Prototypes.Prototype.emg.offline_analysis.peak_classifier import PeakClassifier 
    from offline_analysis.batch_analyzer import BatchAnalyzer
import os
import sys
from pathlib import Path

# Add parent folder (2025) to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from emg_peak_feature_analyser import EMGPeakAnalyzer

def select_csv_file():
    """Select CSV file using PyQt5 file dialog."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select EMG CSV File",
        "",
        "CSV Files (*.csv)"
    )
    return file_path

class AnalysisModeDialog(QMainWindow):
    """PyQt5 dialog for selecting analysis mode."""
    
    def __init__(self):
        super().__init__()
        self.mode = "cancel"
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("EMG Peak Analysis Mode Selection")
        self.setFixedSize(450, 250)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        
        # Center the window
        self.center_window()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("Select Analysis Mode")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Radio buttons group
        group_box = QGroupBox()
        group_layout = QVBoxLayout(group_box)
        
        self.button_group = QButtonGroup()
        
        self.single_radio = QRadioButton("Single File Analysis")
        self.single_radio.setChecked(True)
        self.batch_radio = QRadioButton("Batch Dataset Analysis")
        
        self.button_group.addButton(self.single_radio, 1)
        self.button_group.addButton(self.batch_radio, 2)
        
        group_layout.addWidget(self.single_radio)
        group_layout.addWidget(self.batch_radio)
        
        layout.addWidget(group_box)
        layout.addSpacing(30)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.setFixedWidth(100)
        self.ok_button.clicked.connect(self.on_ok)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedWidth(100)
        self.cancel_button.clicked.connect(self.on_cancel)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
    def center_window(self):
        """Center the window on screen."""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
    def on_ok(self):
        """Handle OK button click."""
        if self.single_radio.isChecked():
            self.mode = "single"
        elif self.batch_radio.isChecked():
            self.mode = "batch"
        self.close()
        
    def on_cancel(self):
        """Handle Cancel button click."""
        self.mode = "cancel"
        self.close()

def select_analysis_mode():
    """Let user choose between single file or batch analysis using PyQt5."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    dialog = AnalysisModeDialog()
    dialog.show()
    app.exec_()
    
    return dialog.mode

def run_single_file_analysis():
    """Run analysis on a single CSV file."""
    print("Please select a CSV file to analyze...")
    file_path = select_csv_file()

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {file_path}")

    sampling_rate = 10000  # Updated to 10kHz to match training database
    height_percentile = 98
    min_distance = 3  

    # Try to find a trained XGBoost model
    model_path = None
    possible_model_paths = [
        Path(__file__).parent.parent / "models" / "xgboost_model.pkl",
        Path(__file__).parent.parent / "xgboost_model.pkl",
        Path(__file__).parent / "xgboost_model.pkl"
    ]
    
    for path in possible_model_paths:
        if path.exists():
            model_path = str(path)
            print(f"Found XGBoost model: {model_path}")
            break
    
    if not model_path:
        print("Warning: No XGBoost model found. Classification will be skipped.")

    analyzer = PeakClassifier(
        csv_path=file_path,
        model_path=model_path,
        sampling_rate=sampling_rate,
        height_percentile=height_percentile,
        min_distance=min_distance
    )

    results = analyzer.run(show_plots=False, save_results=True, classify_peaks=True)

    print("Analysis Complete")
    print(f"Detected {results['num_peaks']} peaks")
    print(f"Duration: {results['signal_duration']:.2f} seconds")
    
    if results.get('classifications'):
        print(f"Classified {len(results['classifications'])} peaks")
        # Show classification summary
        class_counts = {}
        for result in results['classifications']:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("Classification summary:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}% MVC: {count} peaks")

def run_batch_analysis():
    """Run batch analysis on the combined dataset."""
    # Check if default dataset exists
    default_dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex_master.csv"
    
    if default_dataset_path.exists():
        print(f"Found default dataset: {default_dataset_path}")
        dataset_path = default_dataset_path
    else:
        print("Default dataset not found. Please select the combined dataset file...")
        dataset_path = select_csv_file()
        
        if not dataset_path:
            print("No dataset selected. Exiting.")
            return
    
    print(f"Using dataset: {dataset_path}")
    
    # Run batch analysis
    batch_analyzer = BatchAnalyzer(dataset_path)
    batch_analyzer.load_data()
    batch_analyzer.run()

def run_offline_analysis():
    """Main function to run offline EMG analysis."""
    print("=" * 50)
    print("EMG Peak Analysis Tool")
    print("=" * 50)
    
    # Let user choose analysis mode
    mode = select_analysis_mode()
    
    if mode == "cancel":
        print("Analysis cancelled.")
        return
    elif mode == "single":
        run_single_file_analysis()
    elif mode == "batch":
        run_batch_analysis()
    else:
        print(f"Unknown mode: {mode}")

def main():
    """Entry point for standalone execution."""
    run_offline_analysis()

if __name__ == "__main__":
    main()
