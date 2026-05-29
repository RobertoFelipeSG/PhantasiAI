import numpy as np
import os
import sys
import gpiod
from time import sleep, time
from pathlib import Path
from gpiod.line import Direction, Value

from config.connection_manager import logging
from config.config_manager import load_config

CONFIG = load_config()

class Stimulator:
    def __init__(self, profiler, mqtt_client=None, client_topic=None):
        self.profiler = profiler
        self.mqtt_client = mqtt_client
        self.client_topic = client_topic

        self.gpio_chip = CONFIG.get("gpio_chip")
        self.gpio_pin = CONFIG.get("gpio_pin")
        self.default_duration = CONFIG.get("stim_default_duration")
        self.duration_mapping = CONFIG.get("stim_duration_mapping")

    def _run_stimulation(self, curr_trial, best_params, ready_duration=None):
        ''' Runs parameter generation for GPIO square stimulation '''       
        # Define GPIO pin (BCM numbering)
        GPIO_CHIP = self.gpio_chip

        # Open GPIO chip and request the line
        chip = gpiod.Chip(GPIO_CHIP)
        gpio_pin = self.gpio_pin
        if ready_duration is not None:
            duration = self.duration_mapping.get(float(ready_duration), self.default_duration)
        else: duration = self.default_duration
        
        # Define frequency (Hz) and duty cycle (0-1)
        dutycycle = best_params.get('dutycycle')
        frequency = best_params.get('frequency')
        
        total_cycles = int(duration*frequency)
        period = 1/frequency
        on_time = period * (dutycycle)
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
        
        logging.info(f"[Square] Stimulating ({duration}s for {ready_duration}s ISI) at frequency: {frequency}, dutycycle: {dutycycle}")
        try:
            for cycle in range(total_cycles):
                # Set pin HIGH
                if on_time > 0:  # Only set high if duty cycle > 0
                    line_request.set_value(gpio_pin, Value.ACTIVE)
                    sleep(on_time)
                
                # Set pin LOW
                if off_time > 0:  # Only set low if duty cycle < 100
                    line_request.set_value(gpio_pin, Value.INACTIVE)
                    sleep(off_time)
        
        finally:
            line_request.set_value(gpio_pin, Value.INACTIVE) # Ensure pin is set LOW at the end
            line_request.release() # Release the GPIO line
            self.profiler.log_metric(curr_trial, "stim", duration)

    def run(self, best_params, ready_duration, curr_trial):
        #logging.info("[Square] Starting stimulation process...")
        start_time = time()
        
        self._run_stimulation(curr_trial, best_params, ready_duration)

        duration = time() - start_time
        self.profiler.log_metric(curr_trial, "stim_process", duration)
        
        # message= f"[Square] Stimulation completed. Duration: {time() - start_time:.2f} seconds."
        # logging.info(message)