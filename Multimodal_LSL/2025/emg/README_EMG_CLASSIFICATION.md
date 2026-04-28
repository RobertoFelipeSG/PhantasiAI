# EMG Peak Classification System

This document explains how to use the EMG peak classification system that integrates with the main PhantasiAI GUI.

## Overview

The EMG peak classification system allows:
1. Train an LDA model using an EMG database
2. Record EMG data using the main application
3. Automatically classify detected peaks after recording using the trained LDA model
4. Get detailed classification results with confidence scores and enhanced features

## Feature Extraction (v2.0)

The system now extracts 4 features:

1. Peak Amplitude (`fwEMG 3`): Maximum peak amplitude in µV
2. Minimum Peak Amplitude (`Min_Peak_Amplitude`): Minimum peak amplitude in µV
3. Mean Frequency (`Mean_Frequency`): Frequency-weighted average in Hz
4. Median Frequency (`Median_Frequency`): Frequency where cumulative power reaches 50% in Hz

### Frequency Analysis Implementation (v2.6)

- **Primary Library**: NeuroKit2 for enhanced physiological signal analysis
- **Fallback Library**: SciPy for robust frequency analysis when NeuroKit2 fails
- **Method**: 
  - Primary: NeuroKit2 `signal_psd()` for physiological signal processing
  - Fallback: SciPy `signal.welch()` for standard frequency analysis
- **Sampling Rate**: 10kHz (10,000 Hz) - matches the training database
- **Segment Size**: 20ms window centered around each peak (200 samples at 10kHz)
- **Requirements**: NeuroKit2 is preferred, SciPy is required for fallback
- **Error Handling**: Automatic fallback to SciPy when NeuroKit2 fails, NaN values only when both methods fail
- **Note**: NeuroKit2 provides enhanced physiological signal analysis optimized for EMG data, SciPy ensures robustness

### Window Size Optimization

- **Window**: 20ms centered around each peak
- **Rationale**: More focused analysis on the actual peak activity
- **Sample Count**: 200 samples at 10kHz sampling rate
- **Benefits**: 
  - Reduces noise from surrounding signal
  - More precise frequency analysis
  - Faster computation
  - Better peak-specific feature extraction
  - Optimal data length for NeuroKit2 analysis

### Error Handling and Robustness

- **Primary Strategy**: NeuroKit2 for enhanced physiological analysis
- **Fallback Strategy**: Automatic fallback to SciPy when NeuroKit2 fails
- **Final Fallback**: NaN values only when both NeuroKit2 and SciPy fail
- **LDA Handling**: LDA classifier handles remaining NaN values with feature-specific defaults
- **Default Values** (LDA classifier): 
  - Mean Frequency: 50.0 Hz (typical EMG mean frequency)
  - Median Frequency: 45.0 Hz (typical EMG median frequency)
  - Amplitude features: Median of available values or 0.0
- **Robustness**: Ensures maximum feature extraction success while maintaining data integrity


## Prerequisites

Before using peak classification, we need to:

1. **Install NeuroKit2**: Primary library for enhanced frequency analysis
   ```bash
   pip install neurokit2
   ```
2. **Install SciPy**: Required for fallback frequency analysis
   ```bash
   pip install scipy
   ```
3. Train the LDA model: Run the training pipeline to create a model from the EMG database
4. Save the trained model: The script looks for `lda_model.pkl` in the `emg/` directory

## Training the LDA Model (in a computer)

### Step 1: Run Enhanced Peak Analysis 
```bash
cd Multimodal_LSL/2025/emg/
python batch_peak_analysis.py
```

This will:
- Process the EMG database files
- Detect peaks in the EMG signals
- Extract enhanced features (amplitude + frequency)
- Save results as: `dataset/batch_peak_analysis_YYYYMMDD_HHMMSS/peak_analysis_results.csv`

### Step 2: Train Enhanced LDA Classifier
```bash
python emg_LDA_classifier.py
```

This will:
- Load the enhanced peak analysis results
- Train an LDA model using all 4 features
- Save the trained model as `lda_model.pkl`
- Generate classification results and performance metrics

## Using Peak Classification (in the RPi)

### Start the GUI
```bash
cd Multimodal_LSL/2025/
python main.py
```

### Recording and Classification workflow

1. Connect EMG sensor 
2. Start the GUI
3. Start recording EMG values by clicking the "Start Recording" button
4. Perform EMG contractions during recording following the time marker displayed on the interface
5. Stop recording when finished
6. Once a recording is saved, automatic processing occurs:
   - Peak detection on the recorded EMG data
   - Feature extraction (amplitude + frequency)
   - Classification of detected peaks using the LDA model
   - Results are displayed in the chat window

### Classification Results

After the processing is done, in the chat will appear something similar as:
```
Peak analysis completed: 5 peaks detected, 5 classified
Classification summary: 10% MVC: 1, 25% MVC: 2, 50% MVC: 2
```

Results explanation:
- 5 peaks detected: Number of peaks found in the EMG signal
- 5 classified: Number of peaks successfully classified
- Classification summary: Distribution of MVC percentages over the detected peaks 

### Recording Session Files

During recording, the following files are created in `emg-recordings/YYYY-MM-DD_HHhMMmSS/`:

- `emg.csv`: Raw EMG data with timestamps and event marker
- `peaks.txt`: Peak analysis results with classification details
- `peak_classifications.csv`: Detailed classification results in CSV format

#### Classification file format

The `peak_classifications.csv` file now contains:
```csv
peak_id,timestamp,amplitude,min_amplitude,mean_frequency,median_frequency,predicted_class,confidence,prob_10,prob_25,prob_50
1,2.345,85.2,45.1,12.5,8.3,25,0.87,0.05,0.87,0.08
2,4.123,120.5,78.2,15.2,10.1,50,0.92,0.02,0.06,0.92
```

Where:
- peak_id: Sequential peak number
- timestamp: Time of peak occurrence (seconds)
- amplitude: Peak amplitude (µV)
- min_amplitude: Minimum peak amplitude (µV)
- mean_frequency: Mean frequency (Hz)
- median_frequency: Median frequency (Hz)
- predicted_class: Predicted MVC percentage
- confidence: Classification confidence (0-1)
- prob_X: Probability for each MVC class

## Testing the LDA Model 

### Test the Peak Classifier
```bash
cd Multimodal_LSL/2025/emg/
python test_peak_classifier.py
```

This test:
- Loads the trained LDA model with 4 features
- Generates synthetic EMG data with known peaks
- Processes the data through enhanced peak detection and classification
- Verifies that classifications are working correctly with new features

### Expected Output
```
EMG Peak Classifier Test

Testing Model Loading

- Model file found: lda_model.pkl
- Model loaded successfully
- Available classes: [10 25 50]

Testing Standalone Classifier

* Peak classifier initialized successfully
Analysis completed:
  Peaks detected: 4
  Classifications: 4

Testing EMG Peak Classifier

Creating synthetic EMG data...
Test CSV created: /tmp/.../test_emg.csv

Testing peak classification...

* Test Results:
Peaks detected: 4
Classifications: 4
Classifier available: True

Classification Details:
  Peak 1: 2.000s - 10% MVC (Amplitude: 30.1µV, Confidence: 0.92)
  Peak 2: 4.000s - 25% MVC (Amplitude: 80.3µV, Confidence: 0.88)
  Peak 3: 6.000s - 50% MVC (Amplitude: 120.2µV, Confidence: 0.85)
  Peak 4: 8.000s - 50% MVC (Amplitude: 180.1µV, Confidence: 0.91)

All tests passed!
```

## Other ways to use the available codes: Manual classification

### Using the Command Line

With an specific EMG file:
```bash
cd Multimodal_LSL/2025/emg/
python emg_peak_classifier.py path/to/emg.csv
```

With a specific model:
```bash
python emg_peak_classifier.py path/to/emg.csv path/to/model.pkl
```



## LDA Model Configuration

### Model Parameters
The LDA model is trained using:
- Features: peak amplitude, min amplitude, mean frequency, median frequency
- Classes: MVC percentages (10%, 25%, 50%)
- Algorithm: Linear Discriminant Analysis with StandardScaler

### Feature Engineering
- Amplitude Features: Capture signal strength and range
- Frequency Features: Capture signal quality and spectral characteristics
- Preprocessing: StandardScaler normalization for all features
- NaN Handling: handling of missing frequency data

### Peak Detection Parameters
- Height percentile: 98th percentile threshold for peak detection
- Minimum distance: 3 seconds between peaks
- Sampling rate: 220 Hz

### Classification Parameters
- Confidence threshold: All classifications are returned
- Probability output: Full probability distribution for each class
- Feature scaling: Automatic scaling using trained StandardScaler

## Technical Implementation Details

### Enhanced Feature Extraction

The system now uses SciPy for frequency analysis:

```python
# Frequency analysis using SciPy
from scipy import signal
f, Pxx = signal.welch(segment, sampling_rate)

# Mean frequency calculation
mean_freq = np.sum(f * Pxx) / np.sum(Pxx)

# Median frequency calculation
cumsum_power = np.cumsum(Pxx)
median_idx = np.argmin(np.abs(cumsum_power - 0.5 * cumsum_power[-1]))
median_freq = f[median_idx]
```

### Library Dependencies

- SciPy: For frequency analysis (future replacement with PySiology)
- scikit-learn**: For LDA classification



## File Structure

```
Multimodal_LSL/2025/emg/
├── lda_model.pkl                    # Trained LDA model (4 features)
├── emg_peak_classifier.py           # Enhanced peak classification module
├── emg_LDA_classifier.py           # Enhanced LDA training module
├── batch_peak_analysis.py          # Enhanced peak analysis script
├── test_peak_classifier.py         # Test script
└── README_EMG_CLASSIFICATION.md    # This file

emg-recordings/YYYY-MM-DD_HHhMMmSS/
├── emg.csv                         # Raw EMG data
├── peaks.txt                       # Peak analysis + classification results
└── peak_classifications.csv       # Enhanced classification results
```


