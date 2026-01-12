import yaml
from pathlib import Path
from connection_manager import logging

# Save the config file in this module’s directory
CONFIG_DIR  = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "config.yaml"

# default settings: Ganglion Board + RPi500 
DEFAULT_CONFIG = {
    # web connection
    "host_IPv4": "0.0.0.0",
    
    # board setup
    "serial_port": "/dev/ttyACM0",
    "mac_address": None, 
    "num_emg_ch": 1,
    "num_accel_ch": 3,
    "selected_channels": [0],
    
    # data processing
    "data_acq_buffer_seconds": 2,
    "num_points": 40,
    
    # EMG signal filterings
    "cutoff_freq": 50,
    "cutoff_order": 4,
    
    # data recording
    "recorder_buffer_seconds": 120,
    "marker_interval": 6,

    # feature extraction
    "height_percentile": 98,
    "min_distance": 3,
    "feature_cutoff_freq": 80,

    # gpbo
    "iterations": 20,
    "num_rand": 1,
    "kappa": 3.0, 
    "AF_name": "NEI", # "NEI", "EI", "UBC"
    "noise_level": 0.05
}

def load_config():
    """
    Load config from CONFIG_PATH, or fall back to defaults
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.info(f"Error loading YAML: {e}. Using default settings")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """
    Write the given config dict back to CONFIG_PATH.
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, sort_keys=False)
    except Exception as e:
        print(f"Failed to save config: {e}")
