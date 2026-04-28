#!/usr/bin/env python3
"""
Test NeuroKit2 Integration
=========================

This script tests NeuroKit2's EMG frequency analysis capabilities
with the 20ms window approach.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_neurokit2_availability():
    """Test if NeuroKit2 is available and working."""
    print("Testing NeuroKit2 availability...")
    
    try:
        import neurokit2 as nk
        print("✓ NeuroKit2 successfully imported")
        print(f"  Version: {nk.__version__}")
        return True
    except ImportError as e:
        print(f"✗ NeuroKit2 import failed: {e}")
        return False

def create_test_emg_signal():
    """Create a synthetic EMG signal for testing."""
    print("Creating synthetic EMG signal...")
    
    # Parameters
    sampling_rate = 220  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create synthetic EMG signal with known frequency components
    signal = np.random.normal(0, 0.01, len(t))
    
    # Add a peak at 0.5 seconds
    peak_time = 0.5
    peak_idx = int(peak_time * sampling_rate)
    peak_width = int(0.01 * sampling_rate)  # 10ms peak
    
    for i in range(max(0, peak_idx - peak_width), min(len(signal), peak_idx + peak_width)):
        signal[i] = 0.5 * np.exp(-((i - peak_idx) / (peak_width/3))**2)
    
    return t, signal

def test_neurokit2_emg_analysis(segment, sampling_rate):
    """Test NeuroKit2's EMG frequency analysis on the segment."""
    print("Testing NeuroKit2 EMG frequency analysis...")
    
    if not test_neurokit2_availability():
        print("NeuroKit2 not available, skipping test")
        return None, None
    
    try:
        import neurokit2 as nk
        
        print(f"Segment type: {type(segment)}")
        print(f"Segment shape: {segment.shape if hasattr(segment, 'shape') else 'no shape'}")
        print(f"Segment length: {len(segment)}")
        print(f"Segment dtype: {segment.dtype}")
        print(f"First few values: {segment[:5]}")
        
        # Try different NeuroKit2 analysis methods
        methods_to_try = [
            ("signal_psd", "Signal PSD"),
            ("signal_power", "Signal Power"),
            ("signal_spectral", "Signal Spectral")
        ]
        
        for method_name, description in methods_to_try:
            try:
                print(f"\nTrying method: {description} ({method_name})")
                
                if method_name == "signal_psd":
                    # Try signal PSD analysis
                    psd = nk.signal_psd(segment, sampling_rate=sampling_rate)
                    print(f"  PSD result type: {type(psd)}")
                    if hasattr(psd, 'shape'):
                        print(f"  PSD shape: {psd.shape}")
                    if hasattr(psd, 'columns'):
                        print(f"  PSD columns: {psd.columns.tolist()}")
                    
                elif method_name == "signal_power":
                    # Try signal power analysis
                    power = nk.signal_power(segment, sampling_rate=sampling_rate)
                    print(f"  Power result: {power}")
                    
                elif method_name == "signal_spectral":
                    # Try signal spectral analysis
                    spectral = nk.signal_spectral(segment, sampling_rate=sampling_rate)
                    print(f"  Spectral result: {spectral}")
                    
                print(f"  ✓ Method {method_name} executed successfully")
                
            except Exception as e:
                print(f"  ✗ Method {method_name} failed: {e}")
                continue
        
        # Try manual frequency analysis using NeuroKit2's signal processing
        print(f"\nTrying manual frequency analysis...")
        try:
            # Use NeuroKit2's signal processing functions
            # First get the PSD
            psd = nk.signal_psd(segment, sampling_rate=sampling_rate)
            
            print(f"  PSD type: {type(psd)}")
            if hasattr(psd, 'shape'):
                print(f"  PSD shape: {psd.shape}")
            if hasattr(psd, 'columns'):
                print(f"  PSD columns: {psd.columns.tolist()}")
            if hasattr(psd, 'head'):
                print(f"  PSD head: {psd.head()}")
            
            # Check if PSD has data
            if isinstance(psd, pd.DataFrame) and len(psd) > 0:
                # Try to extract frequency and power data
                if 'Frequency' in psd.columns and 'Power' in psd.columns:
                    frequencies = psd['Frequency'].values
                    power = psd['Power'].values
                    
                    # Calculate mean frequency
                    mean_freq = np.sum(frequencies * power) / np.sum(power) if np.sum(power) > 0 else np.nan
                    
                    # Calculate median frequency
                    cumsum_power = np.cumsum(power)
                    if cumsum_power[-1] > 0:
                        median_idx = np.argmin(np.abs(cumsum_power - 0.5 * cumsum_power[-1]))
                        median_freq = frequencies[median_idx]
                    else:
                        median_freq = np.nan
                    
                    print(f"  ✓ Manual analysis successful!")
                    print(f"    Mean Frequency: {mean_freq:.2f} Hz")
                    print(f"    Median Frequency: {median_freq:.2f} Hz")
                    return mean_freq, median_freq
                else:
                    print(f"  ✗ PSD DataFrame missing required columns")
                    print(f"    Available columns: {psd.columns.tolist()}")
            else:
                print(f"  ✗ PSD DataFrame is empty or invalid")
                print(f"    PSD length: {len(psd) if hasattr(psd, '__len__') else 'unknown'}")
                
        except Exception as e:
            print(f"  ✗ Manual analysis failed: {e}")
            return None, None
        
    except Exception as e:
        print(f"NeuroKit2 analysis failed: {e}")
        return None, None

def test_20ms_window_with_neurokit2(t, signal, sampling_rate):
    """Test 20ms window extraction and NeuroKit2 analysis."""
    print("Testing 20ms window with NeuroKit2...")
    
    # Find the peak
    peak_idx = np.argmax(signal)
    peak_time = t[peak_idx]
    
    print(f"Peak found at time: {peak_time:.3f}s (index: {peak_idx})")
    
    # Extract 20ms window
    window_duration = 0.020  # 20 milliseconds
    seg_len = int(window_duration * sampling_rate)
    
    start_idx = max(0, peak_idx - seg_len // 2)
    end_idx = min(len(signal), peak_idx + seg_len // 2)
    segment = signal[start_idx:end_idx]
    
    print(f"Window size: {window_duration*1000:.0f}ms ({seg_len} samples)")
    print(f"Segment length: {len(segment)} samples")
    print(f"Segment time range: {t[start_idx]:.3f}s to {t[end_idx-1]:.3f}s")
    
    # Test NeuroKit2 analysis
    result = test_neurokit2_emg_analysis(segment, sampling_rate)
    if result is not None:
        mean_freq, median_freq = result
    else:
        mean_freq, median_freq = None, None
    
    return mean_freq, median_freq

def test_longer_segments_with_neurokit2(t, signal, sampling_rate):
    """Test NeuroKit2 with longer segments to see if it works better."""
    print("Testing NeuroKit2 with longer segments...")
    
    # Test different segment lengths
    segment_lengths = [50, 100, 200, 500]  # samples
    
    for seg_len in segment_lengths:
        print(f"\nTesting segment length: {seg_len} samples")
        
        # Extract segment from the middle of the signal
        mid_idx = len(signal) // 2
        start_idx = max(0, mid_idx - seg_len // 2)
        end_idx = min(len(signal), mid_idx + seg_len // 2)
        segment = signal[start_idx:end_idx]
        
        print(f"  Segment length: {len(segment)} samples")
        print(f"  Segment time range: {t[start_idx]:.3f}s to {t[end_idx-1]:.3f}s")
        
        # Test NeuroKit2 analysis
        result = test_neurokit2_emg_analysis(segment, sampling_rate)
        if result is not None:
            mean_freq, median_freq = result
            print(f"  ✓ NeuroKit2 worked with {seg_len} samples!")
            print(f"    Mean: {mean_freq:.2f} Hz, Median: {median_freq:.2f} Hz")
            return mean_freq, median_freq
        else:
            print(f"  ✗ NeuroKit2 failed with {seg_len} samples")
    
    print("NeuroKit2 failed with all segment lengths")
    return None, None

def test_scipy_comparison(segment, sampling_rate):
    """Test SciPy for comparison."""
    print("Testing SciPy for comparison...")
    
    try:
        from scipy import signal as scipy_signal
        
        # Calculate frequency features using SciPy
        f, Pxx = scipy_signal.welch(segment, sampling_rate)
        
        # Calculate mean frequency (frequency weighted by power)
        mean_freq = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else np.nan
        
        # Calculate median frequency (frequency where cumulative power is 50%)
        cumsum_power = np.cumsum(Pxx)
        if cumsum_power[-1] > 0:
            median_idx = np.argmin(np.abs(cumsum_power - 0.5 * cumsum_power[-1]))
            median_freq = f[median_idx]
        else:
            median_freq = np.nan
        
        print(f"SciPy results:")
        print(f"  Mean Frequency: {mean_freq:.2f} Hz")
        print(f"  Median Frequency: {median_freq:.2f} Hz")
        
        return mean_freq, median_freq
        
    except Exception as e:
        print(f"SciPy analysis failed: {e}")
        return None, None

def main():
    """Run all tests."""
    print("=" * 60)
    print("NeuroKit2 Integration Test")
    print("=" * 60)
    
    # Test library availability
    neurokit2_available = test_neurokit2_availability()
    
    if not neurokit2_available:
        print("ERROR: NeuroKit2 is not available!")
        print("Please install it with:")
        print("  pip install neurokit2")
        return
    
    # Create test signal
    t, signal = create_test_emg_signal()
    
    print(f"\nSignal statistics:")
    print(f"  Min: {np.min(signal):.6f}")
    print(f"  Max: {np.max(signal):.6f}")
    print(f"  Mean: {np.mean(signal):.6f}")
    print(f"  Std: {np.std(signal):.6f}")
    
    # Test 20ms window with NeuroKit2
    print("\n" + "=" * 40)
    print("20ms Window with NeuroKit2")
    print("=" * 40)
    
    neurokit2_mean, neurokit2_median = test_20ms_window_with_neurokit2(t, signal, 220)
    
    # Test longer segments with NeuroKit2
    print("\n" + "=" * 40)
    print("Longer Segments with NeuroKit2")
    print("=" * 40)
    
    neurokit2_long_mean, neurokit2_long_median = test_longer_segments_with_neurokit2(t, signal, 220)
    
    # Test SciPy for comparison
    print("\n" + "=" * 40)
    print("SciPy Comparison")
    print("=" * 40)
    
    # Extract the same segment for comparison
    peak_idx = np.argmax(signal)
    window_duration = 0.020
    seg_len = int(window_duration * 220)
    start_idx = max(0, peak_idx - seg_len // 2)
    end_idx = min(len(signal), peak_idx + seg_len // 2)
    segment = signal[start_idx:end_idx]
    
    scipy_mean, scipy_median = test_scipy_comparison(segment, 220)
    
    # Compare results
    print("\n" + "=" * 40)
    print("Comparison")
    print("=" * 40)
    
    # Use the longer segment results if available, otherwise use 20ms results
    final_neurokit2_mean = neurokit2_long_mean if neurokit2_long_mean is not None else neurokit2_mean
    final_neurokit2_median = neurokit2_long_median if neurokit2_long_median is not None else neurokit2_median
    
    if final_neurokit2_mean is not None and scipy_mean is not None:
        print("Both methods available - comparing results:")
        print(f"Mean Frequency - NeuroKit2: {final_neurokit2_mean:.2f} Hz, SciPy: {scipy_mean:.2f} Hz")
        print(f"Median Frequency - NeuroKit2: {final_neurokit2_median:.2f} Hz, SciPy: {scipy_median:.2f} Hz")
        
        mean_diff = abs(final_neurokit2_mean - scipy_mean)
        median_diff = abs(final_neurokit2_median - scipy_median)
        
        print(f"Differences - Mean: {mean_diff:.2f} Hz, Median: {median_diff:.2f} Hz")
        
        if mean_diff < 5.0 and median_diff < 5.0:
            print("✓ Results are reasonably consistent")
        else:
            print("⚠ Results show significant differences")
    
    elif final_neurokit2_mean is not None:
        print("✓ Only NeuroKit2 available - using NeuroKit2 results")
    elif scipy_mean is not None:
        print("✓ Only SciPy available - using SciPy results")
    else:
        print("✗ No frequency analysis method available")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
