import RPi.GPIO as GPIO
from time import sleep
import numpy as np
import os
import paho.mqtt.client as mqtt  # Importation du client MQTT

def last_line(n):
    
    num_params = 0
    
    with open('stim.txt', 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while num_params < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b'\n':
                    num_params +=1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return(float(last_line))

if __name__ == "__main__":
    
    
    # Configuration du client MQTT
    broker_address = "localhost"
    client = mqtt.Client("Square_Client")
    client.connect(broker_address, 1883, 60)
    client.loop_start()
    
    # Read parameters
#     with open('stim.txt', 'r') as f:
#         last_lines = f.readlines()[-1]
#     print(last_lines)
    

#     if os.path.exists('hist_params.npy'):
#         try:
#             data = np.load('hist_params.npy', allow_pickle=True)
#             d = data.item()
#             print(f"Stimulation parameters: {data}")
#             k = d.keys()
#             v = list(d.values())
#             print(v, k)
#         except Exception as e:
#             print(f"Error reading {filename}: {e}")
#     else:
#         print(f"Error reading {filename}: {e}")
        
    #GPIO.cleanup()

    # Define GPIO pin (BCM numbering)
    gpio_pin = 12  # Replace with your desired pin (e.g., 18, 23, etc.)

    # Set up GPIO (BCM naming convention)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(gpio_pin, GPIO.OUT)

    # Define frequency (Hz) and duty cycle (0-1)
    frequency =  last_line(2)*10#v[0]*10#35#12000  # Adjust for desired frequency
    duty_cycle = last_line(1)/10#v[1]/10#0.5#0.05 # Adjust for desired duty cycle (0.5 for 50%)
    print('Here are the new stimulation values')
    print(frequency, duty_cycle)
    print('Done!')


    # Run for a certain time (or indefinitely with loop)
    try:
        # Use PWM (Pulse Width Modulation) for square wave generation
        pwm = GPIO.PWM(gpio_pin, frequency)
        pwm.start(duty_cycle * 100)  # Duty cycle as a percentage

        # Replace with your desired runtime (seconds)
        sleep(25)  # Uncomment for 5 seconds runtime
#         while True:
#             pass  # Loop for continuous generation
        #except KeyboardInterrupt:
            #pass  # Handle Ctrl+C interrupt
    finally:
    # Clean up GPIO on exit
        pwm.stop()
        GPIO.cleanup()

    client.loop_stop()
    client.disconnect()

