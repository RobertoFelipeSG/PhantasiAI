import json
from pathlib import Path

# Save the config file in this module’s directory
CONFIG_DIR  = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "phantasiai_config.json"

# default settings: Ganglion Board + RPi500 
DEFAULT_CONFIG = {
    "serial_port": "/dev/ttyACM0",
    "mac_address": None, 
    
    "num_emg_ch": 1,
    "num_accel_ch": 3,
    "selected_channels": [0],
    
    "data_acq_buffer_seconds": 2,
    "num_points": 40,
    
    "cutoff_freq": 50,
    "cutoff_order": 4,
    
    "recorder_buffer_seconds": 120,
    "marker_interval": 6,

    "height_percentile": 98,
    "min_distance": 3,
    "feature_cutoff_freq": 80
}

def load_config():
    """
    Load config from CONFIG_PATH, or fall back to defaults
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """
    Write the given config dict back to CONFIG_PATH.
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")
