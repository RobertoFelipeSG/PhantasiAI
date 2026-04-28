"""
Mock RPi.GPIO module for testing on non-Raspberry Pi systems (like macOS)
This allows the Square.py script to run without actual GPIO hardware.
"""

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
    def PWM(pin, frequency):
        return MockPWM(pin, frequency)

# Create a mock module
import sys
sys.modules['RPi'] = type(sys)('RPi')
sys.modules['RPi.GPIO'] = MockGPIO()
