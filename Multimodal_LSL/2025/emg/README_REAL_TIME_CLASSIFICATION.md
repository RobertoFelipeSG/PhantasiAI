# EMG Peak Classification System

This document explains how to use the EMG peak classification system that integrates with the main PhantasiAI application.

## Overview

The EMG peak classification system allows you to:
1. Train an LDA model using peak analysis results from your EMG database
2. Record EMG data using the main application
3. Automatically classify detected peaks after recording using the trained LDA model
4. Get detailed classification results with confidence scores

## Prerequisites

Before using peak classification, you need to:

1. **Train the LDA Model**: Run the training pipeline to create a model from your EMG database
2. **Ensure Model File Exists**: The system looks for `lda_model.pkl` in the `emg/` directory

## Training the LDA Model

### Step 1: Run Peak Analysis
```bash
cd Multimodal_LSL/2025/emg/
python run_peak_analysis.py
```

This will:
- Process your EMG database files
- Detect peaks in the EMG signals
- Save results to `dataset/batch_peak_analysis_YYYYMMDD_HHMMSS/peak_analysis_results.csv`

### Step 2: Train LDA Classifier
```bash
python run_lda_after_peak_analysis.py
```

This will:
- Load the peak analysis results
- Train an LDA model using peak amplitudes as features
- Save the trained model as `lda_model.pkl`
- Generate classification results and performance metrics

## Using Peak Classification

### Starting the Application
```bash
cd Multimodal_LSL/2025/
python main.py
```

### Recording and Classification Workflow

1. **Connect your EMG sensor** (Arduino or other device)
2. **Start recording** by clicking "Start Recording"
3. **Perform EMG contractions** during recording
4. **Stop recording** when finished
5. **Automatic processing** occurs:
   - Peak detection on the recorded EMG data
   - Classification of detected peaks using the LDA model
   - Results are displayed in the chat window

### Classification Results

After recording stops, you'll see messages like:
```
Peak analysis completed: 5 peaks detected, 5 classified
Classification summary: 10% MVC: 1, 25% MVC: 2, 50% MVC: 2
```

This indicates:
- **5 peaks detected**: Number of peaks found in the EMG signal
- **5 classified**: Number of peaks successfully classified
- **Classification summary**: Distribution of MVC percentages

### Recording Session Files

During recording, the following files are created in `emg-recordings/YYYY-MM-DD_HHhMMmSS/`:

- **`emg.csv`**: Raw EMG data with timestamps
- **`peaks.txt`**: Peak analysis results with classification details
- **`peak_classifications.csv`**: Detailed classification results in CSV format

### Classification CSV Format

The `peak_classifications.csv` file contains:
```csv
peak_id,timestamp,amplitude,predicted_class,confidence,prob_10,prob_25,prob_50
1,2.345,85.2,25,0.87,0.05,0.87,0.08
2,4.123,120.5,50,0.92,0.02,0.06,0.92
```

Where:
- **peak_id**: Sequential peak number
- **timestamp**: Time of peak occurrence (seconds)
- **amplitude**: Peak amplitude (µV)
- **predicted_class**: Predicted MVC percentage
- **confidence**: Classification confidence (0-1)
- **prob_X**: Probability for each MVC class

## Testing the System

### Test the Peak Classifier
```bash
cd Multimodal_LSL/2025/emg/
python test_peak_classifier.py
```

This test:
- Loads the trained LDA model
- Generates synthetic EMG data with known peaks
- Processes the data through peak detection and classification
- Verifies that classifications are working correctly

### Expected Output
```
EMG Peak Classifier Test
============================================================
Testing Model Loading
============================================================
✅ Model file found: lda_model.pkl
✅ Model loaded successfully
Available classes: [10 25 50]

Testing Standalone Classifier
============================================================
✅ Peak classifier initialized successfully
Analysis completed:
  Peaks detected: 4
  Classifications: 4

Testing EMG Peak Classifier
============================================================
Creating synthetic EMG data...
Test CSV created: /tmp/.../test_emg.csv

Testing peak classification...

Test Results:
Peaks detected: 4
Classifications: 4
Classifier available: True

Classification Details:
  Peak 1: 2.000s - 10% MVC (Amplitude: 30.1µV, Confidence: 0.92)
  Peak 2: 4.000s - 25% MVC (Amplitude: 80.3µV, Confidence: 0.88)
  Peak 3: 6.000s - 50% MVC (Amplitude: 120.2µV, Confidence: 0.85)
  Peak 4: 8.000s - 50% MVC (Amplitude: 180.1µV, Confidence: 0.91)

🎉 All tests passed!
```

## Manual Classification

### Using the Command Line
You can also classify EMG recordings manually:

```bash
cd Multimodal_LSL/2025/emg/
python emg_peak_classifier.py path/to/your/emg.csv
```

Or with a specific model:
```bash
python emg_peak_classifier.py path/to/your/emg.csv path/to/your/model.pkl
```

### Using Python Code
```python
from emg.emg_peak_classifier import classify_emg_recording

# Classify an EMG recording
results = classify_emg_recording("path/to/emg.csv", show_plots=True)

# Access results
print(f"Peaks detected: {results['num_peaks']}")
print(f"Classifications: {len(results['classifications'])}")

for result in results['classifications']:
    print(f"Peak {result['peak_id']}: {result['predicted_class']}% MVC")
```

## Configuration

### Model Parameters
The LDA model is trained using:
- **Feature**: Peak amplitude (in µV)
- **Classes**: MVC percentages (e.g., 10%, 25%, 50%)
- **Algorithm**: Linear Discriminant Analysis

### Peak Detection Parameters
- **Height percentile**: 98th percentile threshold for peak detection
- **Minimum distance**: 3 seconds between peaks
- **Sampling rate**: 220 Hz

### Classification Parameters
- **Confidence threshold**: All classifications are returned (no filtering)
- **Probability output**: Full probability distribution for each class

## Troubleshooting

### "No LDA model found" Error
**Solution**: Train the model first using the steps above.

### "Model loading failed" Error
**Solution**: Check that `lda_model.pkl` exists and is not corrupted.

### No Classifications After Recording
**Possible causes**:
- No peaks detected in the EMG signal
- EMG signal is too weak (below detection threshold)
- LDA model is not available

### Poor Classification Accuracy
**Solutions**:
- Retrain the model with more diverse data
- Check that your training data covers the expected MVC range
- Verify that peak detection is working correctly
- Ensure the EMG signal quality is good

## File Structure

```
Multimodal_LSL/2025/emg/
├── lda_model.pkl                    # Trained LDA model
├── emg_peak_classifier.py           # Peak classification module
├── emg_LDA_classifier.py           # LDA training module
├── run_peak_analysis.py            # Peak analysis script
├── run_lda_after_peak_analysis.py  # LDA training script
├── test_peak_classifier.py         # Test script
└── README_REAL_TIME_CLASSIFICATION.md  # This file

emg-recordings/YYYY-MM-DD_HHhMMmSS/
├── emg.csv                         # Raw EMG data
├── peaks.txt                       # Peak analysis + classification results
└── peak_classifications.csv       # Detailed classification results
```

## Advanced Usage

### Custom Model Path
You can specify a custom model path when initializing the classifier:
```python
from emg.emg_peak_classifier import EMGPeakClassifier

classifier = EMGPeakClassifier("emg.csv", model_path="path/to/your/model.pkl")
```

### Adjusting Peak Detection Parameters
```python
classifier = EMGPeakClassifier(
    "emg.csv",
    height_percentile=95,    # Lower threshold
    min_distance=2           # 2 seconds minimum distance
)
```

### Accessing Classification Results
```python
results = classifier.run(classify_peaks=True)

# Get all classifications
classifications = results['classifications']

# Get specific peak classification
peak_1 = classifications[0]
print(f"Peak 1: {peak_1['predicted_class']}% MVC")

# Get confidence scores
confidences = [r['confidence'] for r in classifications]
print(f"Average confidence: {np.mean(confidences):.3f}")
```

## Performance Considerations

- **Processing time**: Peak detection and classification are fast (< 1 second for typical recordings)
- **Memory usage**: Efficient processing with minimal memory overhead
- **Accuracy**: Depends on the quality of the trained model and EMG signal quality
- **File size**: Classification results add minimal overhead to recording files

## Future Enhancements

Potential improvements:
- Support for multiple EMG channels
- Additional features beyond peak amplitude (RMS, frequency features)
- Real-time classification during recording
- Integration with other classification algorithms
- Visualization of classification results
- Model performance monitoring and retraining
