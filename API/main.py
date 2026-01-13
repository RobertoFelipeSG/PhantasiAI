import os
import asyncio
import uvicorn
import paho.mqtt.client as mqtt

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config.connection_manager import manager, logging
from config.config_manager import load_config

from emg.data_acquisition import GanglionData
from emg.synthetic_data_acquisition import SyntheticGanglionData
from stim.change_detector import WatchDog

ganglion_instance = None
watchdog_system = None
mqtt_client = None
CONFIG = load_config()
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mqtt_client, watchdog_system
    
    # MQTT Setup
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "FastAPI_Backend")
        mqtt_client.connect("localhost", 1883, 60) 
        mqtt_client.loop_start()
        logging.info("MQTT Client Connected and loop started")
        
    except Exception as e:
        logging.error(f"Failed to connect to MQTT: {e}") 

    # Setup logic 
    watchdog_system = WatchDog(mqtt_client)

    yield # Server runs HERE

    # Shutdown logic
    if watchdog_system: watchdog_system.stop_watching()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logging.info("Cleanup complete: Watchdog and MQTT stopped")

app = FastAPI(lifespan=lifespan)

script_dir = os.path.dirname(__file__)
static_path = os.path.join(script_dir, "front-end", "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def root():
    '''
    Read the HTML file and return
    '''
    try:
        with open("front-end/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
         return HTMLResponse(content="<h1>Error: front-end/index.html not found!</h1>", status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    '''
    Initialize connection to websocket manager and await data
    '''
    await manager.connect(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logging.info("WebSocket disconnected.")

@app.post("/start_stream")
async def start_stream(num_trials: str, synthetic: bool):
    '''
    Initializes Ganglion
    Starts background EMG thread 
    Start CSV recording of EMG data
    '''
    global ganglion_instance, watchdog_system
    
    if ganglion_instance and ganglion_instance._emg_thread.is_alive(): 
        return {"status": "error", "message": "EMG already streaming"}
    
    # Retrieve reference of event loop running API server
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = asyncio.get_event_loop()
    
    # Create and run EMG execution thread (runs concurrently with FastAPI server)
    if synthetic: 
        ganglion_instance = SyntheticGanglionData(sample_rate=250, num_trials=num_trials)
    else: 
        ganglion_instance = GanglionData(sample_rate=250, num_trials=num_trials)
    
    if watchdog_system:
        directory_to_watch = str(ganglion_instance.recorder.classification_dir)
        watchdog_system.start_watching(directory_to_watch)
    
    ganglion_instance.start(current_loop)
    
    return {"status": "success", "message": "EMG streaming started"}

@app.post("/stop_stream")
async def stop_stream():
    '''
    Stops CSV recording
    Stops background thread
    '''
    global ganglion_instance, watchdog_system

    if watchdog_system:
        watchdog_system.stop_watching()

    if ganglion_instance: # Ensure stopping process is complete
        ganglion_instance.stop() 
        ganglion_instance = None
        return {"status": "success", "message": "EMG streaming stopped"}
    return {"status": "error", "message": "No active stream"}

if __name__ == "__main__":
    logging.info("Starting FastAPI server")
    uvicorn.run(app, host=CONFIG.get('host_IPv4'), port=8000)