# PhantasiAI EMG Recording Application

A Python GUI application for recording and analyzing EMG (Electromyography) signals in real-time.

## Quick Start

### Prerequisites
- Python 3.13+ 
- EMG sensor connected via Arduino
- Windows (tested on Windows 10/11)

### Installation
```bash
# Install required packages
pip install PyQt5 pylsl pyqtgraph scipy numpy matplotlib pandas mne pyfirmata2 pyserial
```

### Configuration
1. Edit `config/phantasiai_config.json`
2. Change `arduino_port` to your COM port (e.g., "COM7")

### Run Application
```bash
python main.py
```

## Features
- **Real-time EMG recording** from multiple channels
- **Live signal visualization** with pyqtgraph
- **Automatic peak detection** when recording stops
- **Machine learning classification** of detected peaks
- **Data export** to CSV format

## Usage
1. Connect EMG sensor to Arduino
2. Update COM port in config file
3. Run `python main.py`
4. Click "Start Recording" to begin
5. Click "Stop Recording" to analyze peaks

## File Structure
- `main.py` - Application entry point
- `widgets/` - GUI components
- `data/` - Data acquisition and processing
- `emg/` - EMG analysis and classification
- `config/` - Configuration files

## Troubleshooting
- **Port not found**: Check Device Manager for correct COM port
- **Import errors**: Install missing packages with pip
- **Arduino issues**: Ensure Firmata sketch is uploaded to Arduino

