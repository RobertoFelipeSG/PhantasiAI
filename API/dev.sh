#!/bin/bash

PORT=8000
PID=$(lsof -t -i:$PORT)

if [ -z "$PID" ]; then
    echo "Port $PORT is clear."
else
    echo "Cleaning up process $PID on port $PORT..."
    kill -9 $PID
    sleep 1
fi

echo "Starting PhantasiAI..."
python3 main.py
