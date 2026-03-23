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
    def __init__(self, profiler, mqtt_client, client_topic):
        self.profiler = profiler
        self.mqtt_client = mqtt_client
        self.client_topic = client_topic

        self.gpio_chip = CONFIG.get("gpio_chip")
        self.gpio_pin = CONFIG.get("gpio_pin")
        self.duration = CONFIG.get("duration")

    def _run_stimulation(self, best_params):
        ''' Runs parameter generation for GPIO square stimulation '''       
        # Define GPIO pin (BCM numbering)
        GPIO_CHIP = self.gpio_chip

        # Open GPIO chip and request the line
        chip = gpiod.Chip(GPIO_CHIP)
        gpio_pin = self.gpio_pin
        duration = self.duration
        
        # Define frequency (Hz) and duty cycle (0-1)
        dutycycle = best_params.get('dutycycle')
        frequency = best_params.get('frequency')
        
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
        
        # Release the GPIO line
        line_request.release()

    def run(self, best_params, curr_trial):
        #logging.info("[Square] Starting stimulation process...")
        start_time = time()
        
        self._run_stimulation(best_params)

        duration = time() - start_time
        self.profiler.log_metric(curr_trial, "stim", duration)
        
        # message= f"[Square] Stimulation completed. Duration: {time() - start_time:.2f} seconds."
        # logging.info(message)