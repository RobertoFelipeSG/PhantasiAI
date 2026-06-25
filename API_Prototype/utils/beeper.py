import math
import wave
import struct
import tempfile
import subprocess
import threading
from config.connection_manager import logging


class Beeper:
    def __init__(
        self,
        enabled=True,
        duration=0.5,
        frequency=1000,
        volume=0.5,
        device="plughw:CARD=vc4hdmi1,DEV=0",
    ):
        self.enabled = enabled
        self.duration = duration
        self.frequency = frequency
        self.volume = volume
        self.device = device
        self._wav_path = self._create_beep_file()

    def _create_beep_file(self):
        sample_rate = 48000
        n_samples = int(sample_rate * self.duration)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        path = tmp.name
        tmp.close()

        with wave.open(path, "w") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)

            for i in range(n_samples):
                value = int(
                    32767
                    * self.volume
                    * math.sin(2 * math.pi * self.frequency * i / sample_rate)
                )
                frame = struct.pack("<hh", value, value)
                wav.writeframes(frame)

        return path

    def beep(self):
        if not self.enabled:
            return

        try:
            subprocess.Popen(
                [
                    "aplay",
                    "-q",
                    "-D",
                    self.device,
                    self._wav_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logging.warning(f"[Beeper] Could not play beep: {e}")