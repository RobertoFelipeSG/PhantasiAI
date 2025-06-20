# emg_features_extractor.py
import numpy as np

class EMGFeatureExtractor:
    """
    Compute time-domain EMG features for multi-channel data:
      - number of channels
      - Mean Absolute Value (MAV)
      - Root Mean Square (RMS)
      - Slope Sign Changes (SSC)
      - Variance
    """

    def __init__(self, npzfile=None, emg_array=None, array_name='emg'):
        if npzfile:
            data = np.load(npzfile, allow_pickle=False)
            if array_name not in data.files:
                raise KeyError(f"Array '{array_name}' not found in {npzfile}.")
            emg = data[array_name]
        elif emg_array is not None:
            emg = np.array(emg_array, copy=False)
        else:
            raise ValueError("Provide either npzfile or emg_array")

        if emg.ndim != 2:
            raise ValueError(f"EMG data must be 2D (time × channels), got shape {emg.shape}")
        self.emg = emg
        self.num_channels = emg.shape[1]
        self._features = None

    @staticmethod
    def _slope_sign_changes(signal):
        diff1 = np.diff(signal)
        return int(np.sum((diff1[:-1] * diff1[1:]) < 0))

    def compute_features(self):
        emg = self.emg
        mav = np.mean(np.abs(emg), axis=0)
        rms = np.sqrt(np.mean(emg**2, axis=0))
        var = np.var(emg, axis=0)
        ssc = np.array([self._slope_sign_changes(emg[:, ch]) 
                        for ch in range(self.num_channels)], dtype=int)
        self._features = {
            'mav': mav,
            'rms': rms,
            'ssc': ssc,
            'var': var
        }
        return self._features

    @property
    def features_dict(self):
        if self._features is None:
            self.compute_features()
        return self._features

    def features_table(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to use features_table()")
        feats = self.features_dict
        df = pd.DataFrame({
            'channel': np.arange(self.num_channels),
            'mav': feats['mav'],
            'rms': feats['rms'],
            'ssc': feats['ssc'],
            'variance': feats['var']
        })
        return df


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Test EMGFeatureExtractor on a .npz in the current dir"
    )
    parser.add_argument(
        "npzfile",
        nargs="?",
        default="example.npz",
        help="Path to EMG .npz file (default: example.npz)"
    )
    args = parser.parse_args()

    try:
        extractor = EMGFeatureExtractor(npzfile=args.npzfile)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded '{args.npzfile}': {extractor.emg.shape[0]} samples × {extractor.num_channels} channels\n")

    feats = extractor.features_dict
    for name, vec in feats.items():
        print(f"{name.upper():3s}: {vec}")

    try:
        df = extractor.features_table()
        print("\nFeature table:")
        print(df.to_string(index=False))
    except ImportError:
        print("\n(pandas not installed; skipping table display)")
