#!/bin/bash
# Script to start MQTT broker for PhantasiAI on Raspberry Pi
echo "Starting MQTT broker for PhantasiAI..."

# Check if mosquitto is already running
if systemctl is-active --quiet mosquitto; then
    echo "MQTT broker (mosquitto) is already running"
else
    echo "Starting MQTT broker..."
    sudo systemctl start mosquitto
    sleep 2
    
    # Verify it's running
    if systemctl is-active --quiet mosquitto; then
        echo "MQTT broker started successfully!"
    else
        echo "Failed to start MQTT broker"
        exit 1
    fi
fi

echo "MQTT broker is ready at localhost:1883"
echo "You can now run your PhantasiAI application with AI and STIM processes."

