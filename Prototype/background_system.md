# Background monitor system execution

Background execution system where AI and STIM processes run invisibly while their output is captured and displayed in the main GUI's chatbot interface and in the main terminal.

## Summary of key changes

### **What I replaced:**
1. Terminal Window Launching 
2. Hardcoded Venv Paths → Dynamic environment detection


### **What I added:**
1. Background process structure: threading, queues, timers
2. Automatic output capture and display 
3. Process lifecycle management: start, monitor, stop
4. MAC Compatibility: mock GPIO, dynamic Python detection
5. MQTT broker setup and management



## Code modifications 

### **1. Main Window (`main_window.py`)**

#### **A. Added required imports**
```python
# Lines 1-7: Added new imports
import os
import sys
import subprocess
import threading  # new
import queue      # new
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
```

#### **B. Added background process state variables**
```python
# Added new instance variables in __init__
# Background processes for AI and STIM
self.ai_process = None                    # Track AI process
self.stim_process = None                  # Track STIM process
self.ai_output_queue = queue.Queue()      # Queue for AI output
self.stim_output_queue = queue.Queue()    # Queue for STIM output
self.output_timer = QtCore.QTimer()       # Timer for processing output
self.output_timer.timeout.connect(self.process_background_output)  # Connect timer
```

#### **C. Replaced terminal launching code**

BEFORE:
```python
# Automatic scripts to run in new terminals
# Find current base directory and thus find the results from the session
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
script_path = os.path.join(base_dir, "stim", "detect_change.py")
script_path_II = os.path.join(base_dir, "stim", "detect_2nd_change.py")
VENV_PATH = "/home/phantasiai/Prototype/prot/bin/activate"

# Target script to be executed in terminals
command_seq = (
f"source {VENV_PATH} &&"
f"echo 'Virtual enviroment actvated' &&"
f"python3 {script_path};"
f"exec bash"
)

command_seq_II = (
f"source {VENV_PATH} &&"
f"echo 'Virtual enviroment actvated' &&"
f"python3 {script_path_II};"
f"exec bash"
)

# The '-e' flag tells lxterminal to execute the command that follows.
command = f'lxterminal -e "bash -c \\"{command_seq}\\""'
command_II = f'lxterminal -e "bash -c \\"{command_seq_II}\\""'

try:
    # Launch the new terminal process without blocking the main script
    subprocess.Popen(command, shell=True)
    print(f"Launched AI in a new LXTerminal window.")
    self.chat.log_event("Acquiring data to execute AI Stimulation Optimization")
    
    subprocess.Popen(command_II, shell=True)
    print(f"Launched STIM in a new LXTerminal window.")
    self.chat.log_event("Sending AI Stimulation")             

except FileNotFoundError:
    print("❌ 'lxterminal' not found.")
    print("Please ensure you're running a version of Raspberry Pi OS with a desktop.")
```

AFTER:
```python
# Start background processes for AI and STIM
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
script_path = os.path.join(base_dir, "stim", "detect_change.py")
script_path_II = os.path.join(base_dir, "stim", "detect_2nd_change.py")

# Start AI process in background
self.chat.log_event(f"Starting AI process: {script_path}")
self.ai_process = self.start_background_process(script_path, "AI", self.ai_output_queue)
if self.ai_process:
    self.chat.log_event("AI Stimulation Optimization started in background")
    print("AI process started in background")
else:
    self.chat.log_event("Failed to start AI process")
    print("Failed to start AI process")

# Start STIM process in background
self.chat.log_event(f"Starting STIM process: {script_path_II}")
self.stim_process = self.start_background_process(script_path_II, "STIM", self.stim_output_queue)
if self.stim_process:
    self.chat.log_event("AI Stimulation started in background")
    print("STIM process started in background")
else:
    self.chat.log_event("Failed to start STIM process")
    print("Failed to start STIM process")

# Start output processing timer
self.output_timer.start(100)  # Check for output every 100ms
```

#### **D. Added background process management methods**

New Method: start_background_process() 
```python
def start_background_process(self, script_path, process_name, output_queue):
    """Start a background process and capture its output."""
    def read_output(process, queue, name):
        """Read output from process and put it in queue."""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    if line:
                        queue.put(f"[{name}] {line}")
                        print(f"[{name}] {line}")  # Also print to main terminal
            process.stdout.close()
        except Exception as e:
            queue.put(f"[{name}] Error reading output: {e}")
            print(f"[{name}] Error reading output: {e}")
    
    try:
        # Try to find and use the current Python environment
        python_executable = sys.executable
        
        # If we're in a virtual environment, use it directly
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            command = f'"{python_executable}" "{script_path}"'
        else:
            # Not in a virtual environment, try to find a common venv path
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            possible_venv_paths = [
                os.path.join(base_dir, "venv", "bin", "activate"),
                os.path.join(base_dir, "env", "bin", "activate"),
                os.path.join(base_dir, ".venv", "bin", "activate"),
                "/home/phantasiai/Prototype/prot/bin/activate"  # Keep original as fallback
            ]
            
            venv_path = None
            for path in possible_venv_paths:
                if os.path.exists(path):
                    venv_path = path
                    break
            
            if venv_path:
                command = f'source "{venv_path}" && python3 "{script_path}"'
                output_queue.put(f"[{process_name}] Using virtual environment: {venv_path}")
            else:
                command = f'python3 "{script_path}"'
                output_queue.put(f"[{process_name}] Using system Python")
        
        output_queue.put(f"[{process_name}] Executing command: {command}")
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Start thread to read output
        output_thread = threading.Thread(
            target=read_output,
            args=(process, output_queue, process_name),
            daemon=True
        )
        output_thread.start()
        
        return process
    except Exception as e:
        output_queue.put(f"[{process_name}] Failed to start: {e}")
        print(f"[{process_name}] Failed to start: {e}")
        return None
```

New Method: stop_background_processes()
```python
def stop_background_processes(self):
    """Stop all background processes."""
    if self.ai_process:
        try:
            self.ai_process.terminate()
            self.ai_process.wait(timeout=5)
        except:
            self.ai_process.kill()
        self.ai_process = None
        
    if self.stim_process:
        try:
            self.stim_process.terminate()
            self.stim_process.wait(timeout=5)
        except:
            self.stim_process.kill()
        self.stim_process = None
```

New Method: process_background_output()
```python
def process_background_output(self):
    """Process output from background processes and display in chat."""
    # Process AI output
    while not self.ai_output_queue.empty():
        try:
            message = self.ai_output_queue.get_nowait()
            self.chat.log_event(message)
        except queue.Empty:
            break
            
    # Process STIM output
    while not self.stim_output_queue.empty():
        try:
            message = self.stim_output_queue.get_nowait()
            self.chat.log_event(message)
        except queue.Empty:
            break
```

#### **E. Modified stop recording**

BEFORE:
```python
if self.recorder.recording:
    # Stop auto analysis timer if running
    if self.auto_analysis_enabled:
        self.stop_auto_analysis_timer()
        # ... rest of stop logic
    self.recorder.stop_recording()
    # ... rest of stop logic
```

AFTER:
```python
if self.recorder.recording:
    # Stop auto analysis timer if running
    if self.auto_analysis_enabled:
        self.stop_auto_analysis_timer()
        self.auto_analysis_enabled = False
        self.btn_auto_analysis.setText("Enable Auto Analysis")
        self.auto_analysis_countdown_label.setText("AI analysis: OFF")
        self.auto_analysis_countdown_label.setStyleSheet("QLabel { color: blue; font-weight: bold; font-size: 15pt; }")
    
    # Stop background processes  
    self.stop_background_processes()  
    self.output_timer.stop()          
    self.chat.log_event("Background AI and STIM processes stopped") 
    
    self.recorder.stop_recording()
    self.btn_record.setText("START")
    
    if self.recorder.session_dir:
        self.chat.log_event(f"Recording saved")
```

#### **F. Modified close event**
BEFORE:
```python
def closeEvent(self, event):
    """Clean up threads/connections on close."""
    # Stop auto analysis timers
    if self.auto_analysis_enabled:
        self.stop_auto_analysis_timer()
    
    if hasattr(self, 'graph') and self.graph.thread.isRunning():
        self.graph.thread.stop()
    if hasattr(self, 'live'):
        self.live.close()
    self.recorder.close()
    super().closeEvent(event)
```

AFTER:
```python
def closeEvent(self, event):
    """Clean up threads/connections on close."""
    # Stop auto analysis timers
    if self.auto_analysis_enabled:
        self.stop_auto_analysis_timer()
    
    # Stop background processes  
    self.stop_background_processes()  
    self.output_timer.stop()          
    
    if hasattr(self, 'graph') and self.graph.thread.isRunning():
        self.graph.thread.stop()
    if hasattr(self, 'live'):
        self.live.close()
    self.recorder.close()
    super().closeEvent(event)
```

### **2. MQTT Scripts - API version**

#### **A. `detect_change.py` **
BEFORE:
```python
mqtt_client = mqtt.Client("FileWatcher_Client")
```

AFTER:
```python
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "FileWatcher_Client")
```

#### **B. `detect_2nd_change.py` (Line 46)**
BEFORE:
```python
mqtt_client = mqtt.Client("SQWatcher_Client")
```

AFTER:
```python
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "SQWatcher_Client")
```

#### **C. `Square.py` (Line 37)**
BEFORE:
```python
client = mqtt.Client("Square_Client")
```

AFTER:
```python
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "Square_Client")
```

#### **D. `gp_code.py` (Line 193)**
BEFORE:
```python
client = mqtt.Client("GPBO_Client")
```

AFTER:
```python
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "GPBO_Client")
```

### **3. BoTorch import fix**

#### **`gp_code.py`**
BEFORE:
```python
from botorch.fit import fit_gpytorch_mll
# ... later 
fit_gpytorch_mll(mll)
```

AFTER:
```python
from botorch.fit import fit_gpytorch_model
# ... later 
fit_gpytorch_model(mll)
```

### **4. Mock GPIO**

#### **`mock_gpio.py`**

```python
class MockPWM:
    def __init__(self, pin, frequency):
        self.pin = pin
        self.frequency = frequency
        self.duty_cycle = 0
        self.running = False
        print(f"[Mock PWM] Created PWM on pin {pin} with frequency {frequency} Hz")
    
    def start(self, duty_cycle):
        self.duty_cycle = duty_cycle
        self.running = True
        print(f"[Mock PWM] Started PWM with {duty_cycle}% duty cycle")
    
    def stop(self):
        self.running = False
        print(f"[Mock PWM] Stopped PWM on pin {self.pin}")
    
    def ChangeDutyCycle(self, duty_cycle):
        self.duty_cycle = duty_cycle
        print(f"[Mock PWM] Changed duty cycle to {duty_cycle}%")
    
    def ChangeFrequency(self, frequency):
        self.frequency = frequency
        print(f"[Mock PWM] Changed frequency to {frequency} Hz")

class MockGPIO:
    BCM = 11
    OUT = 0
    IN = 1
    HIGH = 1
    LOW = 0
    
    @staticmethod
    def setmode(mode):
        print(f"[Mock GPIO] Setting mode: {mode}")
    
    @staticmethod
    def setup(pin, mode):
        print(f"[Mock GPIO] Setting up pin {pin} as {'OUTPUT' if mode == 0 else 'INPUT'}")
    
    @staticmethod
    def output(pin, value):
        print(f"[Mock GPIO] Setting pin {pin} to {'HIGH' if value == 1 else 'LOW'}")
    
    @staticmethod
    def cleanup():
        print("[Mock GPIO] Cleanup called")
    
    @staticmethod
    def PWM(pin, frequency):  # ← NEW METHOD
        return MockPWM(pin, frequency)
```


### **5. New file**

#### **A. `start_mqtt.sh` - MQTT script**
```bash
#!/bin/bash
# Script to start MQTT broker for PhantasiAI on Raspberry Pi
echo "Starting MQTT broker for PhantasiAI..."

# Check if mosquitto is already running
if systemctl is-active --quiet mosquitto; then
    echo "MQTT broker (mosquitto) is already running"
else
    echo "Starting MQTT broker..."
    sudo systemctl start mosquitto
    sleep 2
    
    # Verify it's running
    if systemctl is-active --quiet mosquitto; then
        echo "MQTT broker started successfully!"
    else
        echo "Failed to start MQTT broker"
        exit 1
    fi
fi

echo "MQTT broker is ready at localhost:1883"
echo "You can now run your PhantasiAI application with AI and STIM processes."
```



