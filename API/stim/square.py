import numpy as np
import os
import sys
import gpiod
from time import sleep, time
from pathlib import Path
from config.connection_manager import logging
from gpiod.line import Direction, Value

LOG_FILE = Path(__file__).parent.parent / "test_timings.txt"

class Stimulator:
    def __init__(self, mqtt_client, client_topic):
        self.file_path = None
        self.mqtt_client = mqtt_client
        self.client_topic = client_topic

    def _run_stimulation(self, file_path):
        ''' Runs parameter generation for GPIO square stimulation '''
        self.file_path = file_path
        
        # Define GPIO pin (BCM numbering)
        GPIO_CHIP = "/dev/gpiochip4"

        # Open GPIO chip and request the line
        chip = gpiod.Chip(GPIO_CHIP)
        gpio_pin = 12
        duration = 3
        
        # Define frequency (Hz) and duty cycle (0-1)
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            dutycycle = float(lines[0].strip())
            frequency = float(lines[1].strip())
        
        ''' 
        OLD CODE (with best emg response features in stim.txt):
        max_amp = self._get_line(1)
        min_amp = self._get_line(2)
        denom = max_amp + min_amp
        if denom != 0: # normalized difference between 0 and 100 (eg: 0.5 = 50%)
            duty_cycle = (abs(max_amp - min_amp) / denom) * 100
        else: # avoid dividing by zero
            duty_cycle = 0

        mean_freq = self._get_line(3)
        median_freq = self._get_line(4)
        frequency = (mean_freq + median_freq) / 2 # average of mean and median frequency
        if frequency <= 0: # fail-safe for negative frequency values
            frequency = 1
        '''
        
        logging.info(f"[Square] New stimulation values: frequency: {frequency}, dutycycle: {dutycycle}")
        
        total_cycles = int(duration*frequency)
        period = 1/frequency
        on_time = period * (dutycycle/100)
        off_time = period - on_time
        #print(f"[Square]: {chip}")
    
        line_request = gpiod.request_lines(
            GPIO_CHIP,
            consumer="pwm_generator",
            config={
                gpio_pin: gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    output_value=Value.INACTIVE
                )
            }
        )
        line_request.set_value(gpio_pin, Value.ACTIVE)
        
        for cycle in range(total_cycles):
            # Set pin HIGH
            if on_time > 0:  # Only set high if duty cycle > 0
                line_request.set_value(gpio_pin, Value.ACTIVE)
                sleep(on_time)
            
            # Set pin LOW
            if off_time > 0:  # Only set low if duty cycle < 100
                line_request.set_value(gpio_pin, Value.INACTIVE)
                sleep(off_time)
        
        # Ensure pin is set LOW at the end
        line_request.set_value(gpio_pin, Value.INACTIVE)
        
        logging.info(f"[Square] PWM generation complete!")
        
        # Release the GPIO line
        line_request.release()

    def run(self, file_path):
        #logging.info("[Square] Starting stimulation process...")
        start_time = time()
        
        self._run_stimulation(file_path)
        
        message= f"[Square] Stimulation completed. Duration: {time() - start_time:.2f} seconds."
        logging.info(message)
        
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"{message}\n")
        except OSError as e:
            logging.error(f"Could not write to timing file: {e}")