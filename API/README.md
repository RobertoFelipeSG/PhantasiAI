# PhantasiAI Prototype

This is a testing prototype for an API system to run an EMG-based OpenBCI Ganglion closed-loop optimization system 

## 🗂️ Project Structure

```
PhantasiAI/
├── calibration/
│   └── calibrator.py                   # Calibrator
│   └── calb_change_detector.py         # WatchDogCalb
│   └── create_gt.py                    # GroundTruthGenerator       
├── config/
│   └── config_manager.py               # ConfigManager
│   └── connection_manager.py           # ConnectionManager
├── docs/
│   └── RPI Setup.pdf                   # software setup guide          
├── emg/                 
│   └── data_acquisition.py             # Ganglion
│   └── emg_feature_extractor.py        # FeatureExtractor
│   └── emg_peak_classifier.py          # PeakClassifier
│   └── recorder.py                     # RealTimeRecorder
│   └── synthetic_data_acquisition.py   # SyntheticGanglion
├── front-end/
│   └── static/
│   └── js/      
│   └── index.html
├── stim/
│   └── opt_change_detector.py           # WatchDogOpt
│   └── gbpo_new.py                      # GPBOOptimizer
│   └── square.py                        # Stimulator
├── .gitignore         
├── README.md             
├── dev.sh                               # script to run main.py
├── main.py                              # central API 
└── requirements.txt                     # list of Python dependencies
```

## User Guide (FOLLOW IN-ORDER):
*if you have any problems with these steps check RPi setup file (docs/setup)*
1. Ensure your RPi has been setup as an access point
2. Connect to Internet
3. In PhantasiAI/API/: Create virtual environment and install requirements: pip install -r requirements.txt
4. In PhantasiAI/API/config/: Create config.yaml file (use config.yaml.example as template)
    a. Configure default_serial_port, serial_port_A, selected_channels, gpio_pin, gpio_chip
5. In device home directory: Create setup_phantasiai.sh file (use setup_phantasiai.sh.example as template)
    a. Configure [ACCESS_POINT]
    b. run setup script and reboot RPi
6. [OPTIONAL] For Mosquitto terminal messages run: mosquitto_sub -h localhost -t "emg/#" -v (IN A SEPARATE TERMINAL)
7. Connect to access point (or whichever network you plan to run the application on)
8. On device running application, go to webpage: http://10.42.0.1:8000/
    a. Alternatively, on your RPi device go to webpage: http://127.0.0.1:8000/ (local host)

Server commands (can be run anywhere)
- to stop the application: phstop
- to restart the application: phrestart
- to view real-time logs in terminal: phlogs
- to view the application status: phstatus      

---

## Prototype Architecture Overview

```
[ OpenBCI Ganglion Device ] <-- Catches the raw EMG data
       |                                              
       v 
[ Python Server (main.py) ]  <-- Handles all object/thread/flag creation and controls uvicorn application
       |
[ Ganglion (data_acquisition.py) ]  <-- Catches the raw data from OpenBCI Ganglion device
       |
       +----------------------------------------------+
       |                                              |
  [ Path 1: Visualization ]                (Path 2: Data Processing)
       |                                              | 
  [ WebSockets ]                           [ DataFrames/CSV Files]
       |                                              |
       v                                              v
 [ Browser ]                                 [ Local RPi Device ] 
(Live EMG data/real-time instructions)   (Electrical Stimulation via GPIO)  
```

Data Threads:
1. Main asynchronous thread (FastAPI/Uvicorn): central engine of app instance, controlled by asyncio Event Loop (non-blocking)
2. Log broadcasting: periodically checks ws_handler.log_queue and broadcasts to WebSockets (non-blocking)
3. Websocket endpoint: client-specific thread listening for start/stop commands or pings/pongs from front-end (non-blocking)
4. Heartbeat: client-specific thread that periodically sends pings for connection timeouts (non-blocking)
5. EMG worker: client-specific thread for hardware interaction and data processing, thread=ganglion_instance._emg_thread (non-blocking/blocking)
    - non-blocking during streaming
    - blocking during starting: when self.board_shim.prepare_session() is called (for a maximum of 15 seconds)
    - blocking during stopping: when self._emg_thread.join is called (for a maximum of 5 seconds)
6. Watchdog: client-specific thread to observe changes in features.txt to trigger GPBO (non-blocking)
7. [OPTIONAL] MQTT client: sends messages to seperate terminal (if initialized on server device)
