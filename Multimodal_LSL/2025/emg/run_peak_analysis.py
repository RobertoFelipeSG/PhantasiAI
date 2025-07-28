import tkinter as tk
from tkinter import filedialog
import platform
from emg_peak_analyzer import EMGPeakAnalyzer 

def select_csv_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select EMG CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def main():
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

if __name__ == "__main__":
    main()
