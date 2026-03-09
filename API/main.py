import os
import asyncio
import uvicorn
import json
import paho.mqtt.client as mqtt

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from config.connection_manager import manager, logging
from config.config_manager import load_config

from emg.data_acquisition import GanglionData
from emg.synthetic_data_acquisition import SyntheticGanglionData
from stim.change_detector import WatchDog
from stim.gpbo_new import GPBOOptimizer
from stim.square import Stimulator

mqtt_client = None
CONFIG = load_config()
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mqtt_client
    
    # MQTT Setup (global resource)
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "FastAPI_Backend")
        mqtt_client.connect("localhost", 1883, 60) 
        mqtt_client.loop_start()
        logging.info("[Main] MQTT Client Connected and loop started")
        
    except Exception as e:
        logging.error(f"[Main] Failed to connect to MQTT: {e}") 

    yield # Server runs HERE

    # Shutdown logic
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logging.info("[Main] Cleanup complete: Watchdog and MQTT stopped")

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
    Handles the lifecycle of a client connection
    Listens for commands to Start/Stop streams specific to this client
    '''
    await manager.connect(websocket)
    session = manager.get_session(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()

            try: # get command from frontend (stop or start stream)
                command_data = json.loads(data)
                action = command_data.get("action")

                if action == "start_stream":
                    await handle_start_stream(session, command_data)
                
                elif action == "stop_stream":
                    await handle_stop_stream(session)

            except json.JSONDecodeError:
                logging.error("[Main] Received invalid JSON over WebSocket")
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"[Main] Unexpected error in WebSocket loop: {e}")
        await manager.disconnect(websocket)

async def handle_start_stream(session, data):
    '''
    Initializes Ganglion and Watchdog for client session
    '''
    if session.ganglion and session.ganglion._emg_thread.is_alive(): 
        await session.websocket.send_json({"status": "error", "message": "Stream already running"})
        return
    
    serial_port = data.get("serial_port", CONFIG.get('default_serial_port'))
    num_trials = data.get("num_trials", CONFIG.get('num_trials'))
    is_synthetic = data.get("synthetic", CONFIG.get('synthetic'))
    folder_name = data.get("folder_name", None)
    
    # Retrieve reference of current event loop running API server
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = asyncio.get_event_loop()
    
    # Initialize Ganglion instance
    if is_synthetic: 
        session.ganglion = SyntheticGanglionData(
            websocket=session.websocket,
            serial_port=serial_port,
            sample_rate=250, 
            num_trials=num_trials,
            folder_name=folder_name)
    else: 
        session.ganglion = GanglionData(
            websocket=session.websocket,
            serial_port=serial_port,
            sample_rate=200, 
            num_trials=num_trials,
            folder_name=folder_name)
    
    client_id = id(session)
    
    # Initialize GPBO Optimizer (one per session)
    recordings_directory = str(session.ganglion.recorder.session_dir)
    session.optimizer = GPBOOptimizer(mqtt_client, recordings_directory, client_topic=f"emg/client/{client_id}")

    # Initialize Stimulator
    session.stimulator = Stimulator(mqtt_client, client_topic=f"emg/client/{client_id}")
    
    # Initialize WatchDog instances (using global MQTT client)
    session.watchdog_feat = WatchDog(mqtt_client, client_topic=f"emg/client/{client_id}", change_type='features', optimizer=session.optimizer)
    feat_directory_to_watch = str(session.ganglion.recorder.features_dir)
    session.watchdog_feat.start_watching(feat_directory_to_watch)

    session.watchdog_stim = WatchDog(mqtt_client, client_topic=f"emg/client/{client_id}", change_type='stimulation', stimulator=session.stimulator)
    stim_directory_to_watch = str(Path(__file__).parent / "stim")
    session.watchdog_stim.start_watching(stim_directory_to_watch)
    
    # Start EMG data thread
    session.ganglion.start(current_loop)
    
    logging.info(f"[Main] Started stream for client {client_id}")
    await session.websocket.send_json({"status": "success", "message": "EMG streaming started"})

async def handle_stop_stream(session):
    '''
    Stop Ganglion data thread, Watchdog system, Optimizer and Stimulator for client session
    '''
    
    if session.optimizer:
        session.optimizer.stop()
        session.optimizer = None

    if session.stimulator:
        session.stimulator = None
    
    if session.watchdog_feat:
        session.watchdog_feat.stop_watching()
        session.watchdog_feat = None

    if session.watchdog_stim:
        session.watchdog_stim.stop_watching()
        session.watchdog_stim = None

    if session.ganglion:
        session.ganglion.stop()
        session.ganglion = None
        logging.info(f"[Main] Stopped stream for client {id(session)}")
        await session.websocket.send_json({"status": "success", "message": "EMG streaming stopped"})
    else:
        await session.websocket.send_json({"status": "error", "message": "No active stream to stop"})

if __name__ == "__main__":
    logging.info("[Main] Starting FastAPI server")
    uvicorn.run(app, host=CONFIG.get('host_IPv4'), port=8000)