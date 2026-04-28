
# PhantasiAI - Multimodal LSL 2025

A simple tool for recording and analyzing EMG (muscle) signals from Arduino sensors. Built with PyQt5 for easy use on Windows, Mac, and Linux.

## What Each Package Does

### `main.py`
This is the entry point that starts the entire application. It creates the PyQt5 window, sets up the basic app appearance, applies a small SciPy compatibility patch, and launches the main interface. It decides whether the app runs in live mode (getting data from Arduino) or offline mode (looking at saved files), sets the window size and icon, and cleanly exits when the window closes.

### `widgets/` Package

#### `main_window.py`
This is the central control center for the entire app. It builds the toolbar and layouts, loads settings from the config, wires buttons to actions, and switches between live and offline modes. It passes incoming data to the graph for display, forwards the same data to the recorder when recording, shows basic status in the chat, and handles opening/saving files and reconnecting to the Arduino.

#### `graph_widget.py`
This displays your EMG signals in real-time. It draws a rolling time window with one line per channel, auto-scales the view so signals are readable, and places simple markers where peaks were detected. It receives arrays of timestamps and channel values and focuses only on showing them; it doesn’t save anything.

#### `chat_widget.py`
This shows a log of everything happening in the app. It appends timestamped messages about connection status, recording start/stop, file locations, and analysis summaries, and can export the visible log to a text file when asked.

### `data/` Package

#### `real_time_data.py`
This connects to your Arduino and continuously reads sensor values from up to six analog pins. It looks up the serial port from the config if you don’t provide one, keeps a short buffer of recent samples with timestamps, and exposes those samples to the UI. At the same time it creates an LSL stream per channel so other tools can subscribe to the live data.

### `emg/` Package

#### `emg_recorder.py`
This saves your EMG sessions to disk. When you start, it creates a timestamped folder under `emg-recordings`, opens a CSV, and writes rows with the timestamp, per-channel values (in microvolts), and a simple event flag. It flushes regularly so you don’t lose data, and on stop it closes the file and triggers analysis, then posts a short summary to the chat.

#### `emg_peak_analyzer.py`
This scans a finished recording and finds peaks in the EMG signal. It loads the CSV, runs a few simple detection methods, merges the results, and produces counts and basic stats so you can see how active the signal was over time.

#### `emg_peak_classifier.py`
This takes the detected peaks and groups them into easy-to-read categories. It labels peaks by size and timing pattern and returns a compact summary that can be shown in the chat and saved next to the recording.

### `config/` Package

#### `config_manager.py`
This loads and saves the app’s settings. It reads `phantasiai_config.json` for items like `arduino_port` and default view, provides sensible defaults if the file is missing, and writes changes back when settings are updated.

### `utils/` Package

#### `style.py`
This defines a single stylesheet for the whole app so buttons, labels, and panels look consistent. It’s applied at startup from `main.py`.

#### `path_utils.py`
This helps with safe, cross‑platform file paths and adds the right folders to Python’s module search path so sibling packages can be imported without issues.

#### `scipy_patch.py`
This provides a small shim so the SciPy function the app uses behaves the same across versions, avoiding crashes and keeping analysis and graphs working.

## How Data Moves Through the App

1. **Arduino** sends sensor readings to `real_time_data.py`
2. **`real_time_data.py`** collects the data and sends it to `graph_widget.py` and `emg_recorder.py`
3. **`graph_widget.py`** shows the data in real-time
4. **`emg_recorder.py`** saves the data to files
5. **`emg_peak_analyzer.py`** and **`emg_peak_classifier.py`** analyze the saved data
6. **`chat_widget.py`** shows what's happening at each step




## 🚀 Getting Started (Everything in `Multimodal_LSL/2025`)

All setup and execution happens inside the `Multimodal_LSL/2025/` folder.

---

### 🐧 Raspberry Pi / Linux

#### 1. Install Python 3.10

Run these commands in the terminal (not Python shell):

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
tar -xf Python-3.10.13.tgz
cd Python-3.10.13
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```

Check the version:

```bash
python3.10 --version
```

---

#### 2. Set up the environment



```bash
cd Multimodal_LSL
cd 2025
python3.10 -m venv .venv
source .venv/bin/activate

please comment the pyqt libraries on the requirements.txt

command to install pyqt on venv 
pip install pyqt5 --config-settings --confirm-license= --verbose



pip install -r requirements.txt
python3 main.py
```

---

### 🪟 Windows

#### 1. Install Python 3.10

- Download the installer:  
  https://www.python.org/downloads/release/python-31013/
- ✅ Be sure to check **“Add Python to PATH”** during installation.
- Install normally.

---

#### 2. Set up the environment

```cmd
cd Multimodal_LSL
cd 2025
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

### 🍎 macOS

#### 1. Install Python 3.10 via Homebrew

```bash
brew install python@3.10
brew link --overwrite python@3.10
```

---

#### 2. Set up the environment

```bash
cd Multimodal_LSL/2025
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

