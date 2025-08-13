#!/usr/bin/env python3
"""
Test PySiology Integration
==========================

This script tests the PySiology integration with the new 20ms window approach
to ensure it works correctly before running the full batch analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_pysiology_availability():
    """Test if PySiology is available and working."""
    print("Testing PySiology availability...")
    
    try:
        from pysiology.electromyography import getMNF, getMDF
        print("✓ PySiology successfully imported")
        return True
    except ImportError as e:
        print(f"✗ PySiology import failed: {e}")
        return False

def test_scipy_availability():
    """Test if SciPy is available as fallback."""
    print("Testing SciPy availability...")
    
    try:
        from scipy import signal
        print("✓ SciPy successfully imported")
        return True
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False

def create_test_emg_signal():
    """Create a synthetic EMG signal for testing."""
    print("Creating synthetic EMG signal...")
    
    # Parameters
    sampling_rate = 220  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create synthetic EMG signal with known frequency components
    # Add some noise and a peak
    signal = np.random.normal(0, 0.01, len(t))
    
    # Add a peak at 0.5 seconds
    peak_time = 0.5
    peak_idx = int(peak_time * sampling_rate)
    peak_width = int(0.01 * sampling_rate)  # 10ms peak
    
    for i in range(max(0, peak_idx - peak_width), min(len(signal), peak_idx + peak_width)):
        signal[i] = 0.5 * np.exp(-((i - peak_idx) / (peak_width/3))**2)
    
    return t, signal

def test_different_window_sizes(t, signal, sampling_rate):
    """Test different window sizes to find what PySiology requires."""
    print("Testing different window sizes...")
    
    # Find the peak
    peak_idx = np.argmax(signal)
    peak_time = t[peak_idx]
    
    print(f"Peak found at time: {peak_time:.3f}s (index: {peak_idx})")
    
    # Test different window sizes
    window_sizes = [0.020, 0.050, 0.100, 0.200, 0.500, 1.000]  # 20ms to 1s
    
    for window_duration in window_sizes:
        seg_len = int(window_duration * sampling_rate)
        
        start_idx = max(0, peak_idx - seg_len // 2)
        end_idx = min(len(signal), peak_idx + seg_len // 2)
        segment = signal[start_idx:end_idx]
        
        print(f"\nTesting {window_duration*1000:.0f}ms window ({seg_len} samples):")
        print(f"  Segment length: {len(segment)} samples")
        
        # Test PySiology with this window size
        try:
            from pysiology.electromyography import getMNF, getMDF, getPSD
            
            # Step 1: Calculate PSD
            segment_list = segment.tolist()
            psd, frequencies = getPSD(segment_list, sampling_rate)
            
            # Step 2: Calculate frequencies
            mean_freq = getMNF(psd, frequencies)
            median_freq = getMDF(psd, frequencies)
            
            if mean_freq is not None and not np.isnan(mean_freq):
                print(f"  ✓ PySiology works with {window_duration*1000:.0f}ms window")
                print(f"    Mean: {mean_freq:.2f} Hz, Median: {median_freq:.2f} Hz")
                return window_duration, segment, mean_freq, median_freq
            else:
                print(f"  ✗ PySiology returned NaN with {window_duration*1000:.0f}ms window")
                
        except Exception as e:
            print(f"  ✗ PySiology failed with {window_duration*1000:.0f}ms window: {e}")
            continue
    
    print("PySiology failed with all window sizes")
    return None, None, None, None

def test_pysiology_frequency_analysis(segment, sampling_rate):
    """Test PySiology frequency analysis on the segment."""
    print("Testing PySiology frequency analysis...")
    
    if not test_pysiology_availability():
        print("PySiology not available, skipping test")
        return None, None
    
    try:
        from pysiology.electromyography import getMNF, getMDF, getPSD
        
        print(f"Segment type: {type(segment)}")
        print(f"Segment shape: {segment.shape if hasattr(segment, 'shape') else 'no shape'}")
        print(f"Segment length: {len(segment)}")
        print(f"Segment dtype: {segment.dtype}")
        print(f"First few values: {segment[:5]}")
        
        # Step 1: Calculate PSD using PySiology's getPSD
        print("\nStep 1: Calculating PSD with PySiology...")
        try:
            # Convert segment to list as expected by getPSD
            segment_list = segment.tolist()
            psd, frequencies = getPSD(segment_list, sampling_rate)
            
            print(f"PSD calculated successfully")
            print(f"  PSD length: {len(psd)}")
            print(f"  Frequencies length: {len(frequencies)}")
            print(f"  Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")
            
            # Step 2: Calculate Mean Frequency using PSD
            print("\nStep 2: Calculating Mean Frequency...")
            mean_freq = getMNF(psd, frequencies)
            print(f"  Mean Frequency: {mean_freq:.2f} Hz")
            
            # Step 3: Calculate Median Frequency using PSD
            print("\nStep 3: Calculating Median Frequency...")
            median_freq = getMDF(psd, frequencies)
            print(f"  Median Frequency: {median_freq:.2f} Hz")
            
            # Check if we got valid results
            if mean_freq is not None and not np.isnan(mean_freq) and median_freq is not None and not np.isnan(median_freq):
                print(f"\n✓ PySiology analysis successful!")
                print(f"PySiology results:")
                print(f"  Mean Frequency: {mean_freq:.2f} Hz")
                print(f"  Median Frequency: {median_freq:.2f} Hz")
                return mean_freq, median_freq
            else:
                print(f"\n✗ PySiology returned invalid results")
                return None, None
                
        except Exception as e:
            print(f"PySiology PSD calculation failed: {e}")
            return None, None
        
    except Exception as e:
        print(f"PySiology analysis failed: {e}")
        return None, None

def test_scipy_frequency_analysis(segment, sampling_rate):
    """Test SciPy frequency analysis on the segment."""
    print("Testing SciPy frequency analysis...")
    
    if not test_scipy_availability():
        print("SciPy not available, skipping test")
        return None, None
    
    try:
        from scipy import signal
        
        # Calculate frequency features using SciPy
        f, Pxx = signal.welch(segment, sampling_rate)
        
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

def inspect_pysiology_functions():
    """Inspect PySiology function signatures to understand their requirements."""
    print("Inspecting PySiology function signatures...")
    
    try:
        from pysiology.electromyography import getMNF, getMDF, getPSD
        import inspect
        
        print(f"getMNF signature: {inspect.signature(getMNF)}")
        print(f"getMDF signature: {inspect.signature(getMDF)}")
        print(f"getPSD signature: {inspect.signature(getPSD)}")
        
        # Try to get docstrings
        print(f"\ngetMNF docstring:")
        print(getMNF.__doc__)
        
        print(f"\ngetMDF docstring:")
        print(getMDF.__doc__)
        
        return True
    except Exception as e:
        print(f"Error inspecting PySiology functions: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PySiology Integration Test")
    print("=" * 60)
    
    # Test library availability
    pysiology_available = test_pysiology_availability()
    scipy_available = test_scipy_availability()
    
    if not pysiology_available and not scipy_available:
        print("ERROR: Neither PySiology nor SciPy is available!")
        print("Please install at least one of them:")
        print("  pip install pysiology")
        print("  pip install scipy")
        return
    
    # Inspect PySiology functions if available
    if pysiology_available:
        print("\n" + "=" * 40)
        print("PySiology Function Inspection")
        print("=" * 40)
        inspect_pysiology_functions()
    
    # Create test signal
    t, signal = create_test_emg_signal()
    
    print(f"\nSignal statistics:")
    print(f"  Min: {np.min(signal):.6f}")
    print(f"  Max: {np.max(signal):.6f}")
    print(f"  Mean: {np.mean(signal):.6f}")
    print(f"  Std: {np.std(signal):.6f}")
    
    # Test different window sizes to find what PySiology requires
    print("\n" + "=" * 40)
    print("Testing Different Window Sizes")
    print("=" * 40)
    
    working_window, working_segment, pysiology_mean, pysiology_median = test_different_window_sizes(t, signal, 220)
    
    # Test frequency analysis with full signal
    print("\n" + "=" * 40)
    print("Frequency Analysis Tests (Full Signal)")
    print("=" * 40)
    
    # Test PySiology with full signal
    pysiology_mean, pysiology_median = test_pysiology_frequency_analysis(signal, 220)
    
    # Test SciPy with full signal
    scipy_mean, scipy_median = test_scipy_frequency_analysis(signal, 220)
    
    # Compare results
    print("\n" + "=" * 40)
    print("Comparison")
    print("=" * 40)
    
    if pysiology_mean is not None and scipy_mean is not None:
        print("Both methods available - comparing results:")
        print(f"Mean Frequency - PySiology: {pysiology_mean:.2f} Hz, SciPy: {scipy_mean:.2f} Hz")
        print(f"Median Frequency - PySiology: {pysiology_median:.2f} Hz, SciPy: {scipy_median:.2f} Hz")
        
        mean_diff = abs(pysiology_mean - scipy_mean)
        median_diff = abs(pysiology_median - scipy_median)
        
        print(f"Differences - Mean: {mean_diff:.2f} Hz, Median: {median_diff:.2f} Hz")
        
        if mean_diff < 5.0 and median_diff < 5.0:
            print("✓ Results are reasonably consistent")
        else:
            print("⚠ Results show significant differences")
    
    elif pysiology_mean is not None:
        print("✓ Only PySiology available - using PySiology results")
    elif scipy_mean is not None:
        print("✓ Only SciPy available - using SciPy results")
    else:
        print("✗ No frequency analysis method available")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
