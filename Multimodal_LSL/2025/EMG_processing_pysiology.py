import numpy as np
import pandas as pd
import time
import neurokit2 as nk
import scipy.signal

# replaced deprecated hanning with hann
_original_welch = scipy.signal.welch
def patched_welch(*args, **kwargs):
    if 'window' in kwargs and kwargs['window'] == 'hanning':
        kwargs['window'] = 'hann'
    return _original_welch(*args, **kwargs)
scipy.signal.welch = patched_welch

from pysiology.electromyography import (
    getMAV, getRMS, getWL, getZC, getIEMG, getWAMP,
    getVAR, getLOG, getPSD, getMNF, getMDF
)

class EMGFeaturePipeline:
    def __init__(self, file_path, from_npz=False, array_name='emg', fs=2000, num_segments=10):
        self.file_path = file_path
        self.from_npz = from_npz
        self.array_name = array_name
        self.fs = fs
        self.num_segments = num_segments
        self.signal = self._load_emg_data()

    def _load_emg_data(self):
        if self.from_npz:
            data = np.load(self.file_path)
            if self.array_name not in data.files:
                print(f"'{self.array_name}' not found in file. Using first available array: {data.files[0]}")
                self.array_name = data.files[0]
            signal = data[self.array_name]
            if signal.ndim != 1:
                signal = signal[:, 0]
        else:
            df = pd.read_csv(self.file_path, sep="\t")
            df["minute"] = (df["Timestamp"] // 60).astype(int)
            last_minute = df["minute"].max()
            signal = df[df["minute"] == last_minute]["EMG1"].values

        #scaling based on signal magnitude
        max_val = np.abs(signal).max()
        if max_val < 0.01:
            signal *= 1e6
            print("ℹSignal scaled from volts to µV.")
        elif max_val > 1e6:
            signal /= 1e3
            print("Signal values very large — scaling down for safety.")

        print(f"Signal range: min={signal.min():.2f}, max={signal.max():.2f}")
        signal = np.clip(signal, -1e6, 1e6)
        return signal

    def _preprocess_signal(self, signal):
        clean = nk.emg_clean(signal, sampling_rate=self.fs)
        rect = np.abs(clean)
        envelope = nk.signal_smooth(rect)
        amplitude = nk.emg_amplitude(clean)
        return pd.DataFrame({
            "EMG_Raw": signal,
            "EMG_Clean": clean,
            "EMG_Rect": rect,
            "EMG_Envelope": envelope,
            "EMG_Amplitude": amplitude,
        })

    def _extract_features(self, signal, threshold=1e-4):
        seg_len = len(signal) // self.num_segments
        features = {
            key: [] for key in [
                "MAV", "RMS", "MeanFreq", "MedianFreq",
                "WL", "ZC", "IEMG", "WAMP", "VAR", "LogD"
            ]
        }
        #tried modifying logD to avoid crashes and overflow errors
        for i in range(self.num_segments):
            seg = signal[i * seg_len:(i + 1) * seg_len]
            seg_list = list(seg)
            features["MAV"].append(getMAV(seg_list))
            features["RMS"].append(getRMS(seg_list))
            features["WL"].append(getWL(seg_list))
            features["ZC"].append(getZC(seg_list, threshold))
            features["IEMG"].append(getIEMG(seg_list))
            features["WAMP"].append(getWAMP(seg_list, threshold))
            features["VAR"].append(getVAR(seg_list))

            try:
                val = getLOG(seg_list)
                if val == float("inf") or val > 1e308:
                    raise OverflowError
                features["LogD"].append(val)
            except OverflowError:
                print(f"LogD overflow on segment {i + 1}. Replacing with 0.")
                features["LogD"].append(0.0)
            except Exception as e:
                print(f"LogD error on segment {i + 1}: {e}")
                features["LogD"].append(0.0)

            psd, freqs = getPSD(seg_list, self.fs)
            features["MeanFreq"].append(getMNF(psd, freqs))
            features["MedianFreq"].append(getMDF(psd, freqs))

        return features

    def run(self, selected_features, output_path, interaction_sign=1, interaction_scale=0.5):
        processed = self._preprocess_signal(self.signal)
        filtered = processed["EMG_Clean"].values
        full_features = self._extract_features(filtered)
        filtered_features = {k: full_features[k] for k in selected_features}

        vector = [list(filtered_features.values())]
        n_dim = len(selected_features)
        n_val = self.num_segments

        titles = [f"{k} ({self._unit(k)})" for k in selected_features]
        self._save_to_file(vector, titles, 1, n_dim, n_val, interaction_sign, interaction_scale, output_path)
        print(f"Features saved to: {output_path}")

    def _save_to_file(self, vector, titles, n_suj, n_dim, n_val, interaction_sign, interaction_scale, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Sujet;" + ";".join(titles) + ";Interaction_Scale;Interaction_Sign\n")
            for s in range(n_suj):
                f.write(str(s) + ";")
                for d in range(n_dim):
                    f.write(" ".join([f"{round(vector[s][d][v], 2)}" for v in range(n_val)]) + ";")
                f.write(f"{interaction_scale};{interaction_sign}\n")

    def _unit(self, feature_name):
        return {
            "MAV": "µV", "RMS": "µV", "MeanFreq": "Hz", "MedianFreq": "Hz",
            "WL": "µV", "ZC": "count", "IEMG": "µV", "WAMP": "count",
            "VAR": "µV²", "LogD": "µV"
        }.get(feature_name, "")


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select EMG File (.npz or .txt)",
        filetypes=[("EMG data files", "*.npz *.txt")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        exit()

    from_npz = file_path.lower().endswith(".npz")

    start = time.time()

    extractor = EMGFeaturePipeline(
        file_path=file_path,
        from_npz=from_npz
    )

    features_to_extract = [
        "MAV", "RMS", "MeanFreq", "MedianFreq",
        "WL", "ZC", "IEMG", "WAMP", "VAR", "LogD"
    ]

    extractor.run(
        selected_features=features_to_extract,
        output_path="output_features.txt",
        interaction_sign=1,
        interaction_scale=0.5
    )

    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")
