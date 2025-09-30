import json
from pathlib import Path

# Save the config file in this module’s directory
CONFIG_DIR  = Path(__file__).parent
CONFIG_PATH = CONFIG_DIR / "phantasiai_config.json"

DEFAULT_CONFIG = {
    "view_mode": "chat",         # chat or graph
    "arduino_port": "/dev/ttyUSB0"       # default Arduino port
}

def load_config():
    """
    Load config from CONFIG_PATH, or fall back to defaults.
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
