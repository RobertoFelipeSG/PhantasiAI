import tkinter as tk
from tkinter import filedialog, messagebox
import platform
from emg_peak_analyzer import EMGPeakAnalyzer 
from batch_peak_analysis import BatchPeakAnalyzer
import os
import sys
from pathlib import Path

# Add parent folder (2025) to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from emg_peak_feature_analyser import EMGPeakAnalyzer

def select_csv_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select EMG CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def select_analysis_mode():
    """Let user choose between single file or batch analysis."""
    root = tk.Tk()
    root.title("EMG Peak Analysis Mode Selection")
    root.geometry("400x200")
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    mode_var = tk.StringVar(value="single")
    
    def on_ok():
        root.quit()
        root.destroy()
    
    def on_cancel():
        mode_var.set("cancel")
        root.quit()
        root.destroy()
    
    # Title
    title_label = tk.Label(root, text="Select Analysis Mode", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    # Radio buttons
    single_radio = tk.Radiobutton(root, text="Single File Analysis", variable=mode_var, value="single")
    single_radio.pack(pady=5)
    
    batch_radio = tk.Radiobutton(root, text="Batch Dataset Analysis", variable=mode_var, value="batch")
    batch_radio.pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
    ok_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
    cancel_button.pack(side=tk.LEFT, padx=10)
    
    root.mainloop()
    return mode_var.get()

def run_single_file_analysis():
    """Run analysis on a single CSV file."""
    print("Please select a CSV file to analyze...")
    file_path = select_csv_file()

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"✅ Selected file: {file_path}")

    sampling_rate = 220
    height_percentile = 98
    min_distance = 3  

    analyzer = EMGPeakAnalyzer(
        csv_path=file_path,
        sampling_rate=sampling_rate,
        height_percentile=height_percentile,
        min_distance=min_distance
    )

    results = analyzer.run(show_plots=False, save_results=True)

    print("Analysis Complete")
    print(f"Detected {results['num_peaks']} peaks")
    print(f"Duration: {results['signal_duration']:.2f} seconds")

def run_batch_analysis():
    """Run batch analysis on the combined dataset."""
    # Check if default dataset exists
    default_dataset_path = Path(__file__).parent.parent / "dataset" / "combined_emg_dorsiflex.csv"
    
    if default_dataset_path.exists():
        print(f"Found default dataset: {default_dataset_path}")
        dataset_path = default_dataset_path
    else:
        print("Default dataset not found. Please select the combined dataset file...")
        dataset_path = select_csv_file()
        
        if not dataset_path:
            print("No dataset selected. Exiting.")
            return
    
    print(f"✅ Using dataset: {dataset_path}")
    
    # Run batch analysis
    batch_analyzer = BatchPeakAnalyzer(dataset_path)
    batch_analyzer.load_dataset()
    batch_analyzer.run_batch_analysis()

def main():
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

if __name__ == "__main__":
    main()
