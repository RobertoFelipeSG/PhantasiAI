import os
import asyncio
import uvicorn
import json
import time
import threading
import paho.mqtt.client as mqtt

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from config.connection_manager import manager, logging
from config.config_manager import load_config

from config.session_profiler import SessionProfiler
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
    
    folder_name = data.get("folder_name", None)
    serial_port = data.get("serial_port", CONFIG.get('default_serial_port'))
    is_synthetic = data.get("synthetic", CONFIG.get('synthetic'))
    num_trials = data.get("num_trials", CONFIG.get('num_trials'))
    n_iters = data.get("num_iters", CONFIG.get('iterations'))
    n_reps = data.get("num_reps", CONFIG.get('repetitions'))
    
    # Retrieve reference of current event loop running API server
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = asyncio.get_event_loop()
    
    client_id = id(session) # current session client ID for mqtt handling
    shared_dorsi_flag = threading.Event()

    # Create session profiler
    session.profiler = SessionProfiler()
    profiler = session.profiler # current session profiler to store timing metrics
    
    # Initialize Ganglion instance
    if is_synthetic: 
        session.ganglion = SyntheticGanglionData(
            websocket=session.websocket,
            profiler=profiler,
            dorsi_flag=shared_dorsi_flag,
            serial_port=serial_port,
            sample_rate=250, 
            num_trials=num_trials,
            folder_name=folder_name
        )
    else: 
        session.ganglion = GanglionData(
            websocket=session.websocket,
            profiler=profiler,
            dorsi_flag=shared_dorsi_flag,
            serial_port=serial_port,
            sample_rate=200, 
            num_trials=num_trials,
            folder_name=folder_name
        )
    
    session.session_dir = str(session.ganglion.recorder.session_dir) # current session data folder (initialized within recorder)
    
    # Initialize Stimulator and GPBO Optimizer
    def trigger_auto_stop(): # callback function to trigger auto-stop from GPBO
        asyncio.run_coroutine_threadsafe(handle_stop_stream(session), current_loop)
    
    session.stimulator = Stimulator(profiler, mqtt_client, client_topic=f"emg/client/{client_id}")
    session.optimizer = GPBOOptimizer(session.stimulator, shared_dorsi_flag, n_iters, n_reps, 
                                      session.session_dir, profiler, mqtt_client, 
                                      client_topic=f"emg/client/{client_id}",
                                      on_complete=trigger_auto_stop)
    
    # Initialize WatchDog instance
    session.watchdog_feat = WatchDog(profiler, mqtt_client, client_topic=f"emg/client/{client_id}", 
                                     optimizer=session.optimizer)
    feat_directory_to_watch = str(session.ganglion.recorder.features_dir)
    session.watchdog_feat.start_watching(feat_directory_to_watch)
    
    # Start EMG data thread
    session.ganglion.start(current_loop)

    start_time = time.time()
    while time.time() - start_time < 15:
        if session.ganglion.connection_status == "connected":
            await session.websocket.send_json({"status": "success", "message": "EMG streaming started"})
            return
        
        if session.ganglion.connection_status == "failed":
            break

        await asyncio.sleep(0.1)

    # if no successful connection within 15 seconds, assume failure
    await session.websocket.send_json({"status": "error", "message": "Failed to start EMG stream"})

async def handle_stop_stream(session):
    '''
    Stop Ganglion data thread, Watchdog system, Optimizer and Stimulator for client session
    '''
    if session.watchdog_feat:
        session.watchdog_feat.stop_watching()
        session.watchdog_feat = None
    
    if session.optimizer:
        session.optimizer.handle_stop()
        session.optimizer = None

    if session.stimulator:
        session.stimulator = None

    if session.ganglion:
        session.ganglion.stop()
        session.ganglion = None
        logging.info(f"[Main] Stopped stream for client {id(session)}")
        
        if session.profiler: # export timings once everything has shutdown
            csv_path = session.profiler.save_as_csv(Path(session.session_dir))
            session.profiler = None
        
        await session.websocket.send_json({"status": "success", "message": "EMG streaming stopped"})
    else:
        await session.websocket.send_json({"status": "error", "message": "No active stream to stop"})

if __name__ == "__main__":
    logging.info("[Main] Starting FastAPI server")
    uvicorn.run(app, host=CONFIG.get('host_IPv4'), port=8000)