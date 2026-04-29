import os
import queue
import shutil
import asyncio
import uvicorn
import json
import time
import threading
import paho.mqtt.client as mqtt

from starlette.websockets import WebSocketState
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from config.connection_manager import manager, logging, ws_handler
from config.config_manager import load_config

from config.session_profiler import SessionProfiler
from emg.data_acquisition import GanglionData
from emg.synthetic_data_acquisition import SyntheticGanglionData
from stim.opt_change_detector import WatchDogOpt
from stim.gpbo import GPBOOptimizer
from stim.square import Stimulator
from calibrate.calb_change_detector import WatchDogCalb
from calibrate.calibrator import Calibrator
from calibrate.create_gt import GTGenerator

mqtt_client = None
CONFIG = load_config()
    
async def log_broadcaster():
    while True:
        while not ws_handler.log_queue.empty():
            try:
                msg = ws_handler.log_queue.get_nowait()
                await manager.broadcast(msg)
            except queue.Empty: break
        await asyncio.sleep(0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log Broadcaster Setup (global resource)
    broadcaster = asyncio.create_task(log_broadcaster())
    
    # MQTT Setup (global resource)
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "FastAPI_Backend")
        mqtt_client.connect("localhost", 1883, 60) 
        mqtt_client.loop_start()
        logging.info("[Main] MQTT Client Connected and loop started")
    except Exception as e:
        logging.error(f"[Main] Failed to connect to MQTT: {e}") 

    yield # Server runs HERE

    # Shutdown logic
    for ws in list(manager.active_connections.keys()):
        session = manager.active_connections[ws]
        
        if session.watchdog:
            session.watchdog.stop_watching()
        if session.optimizer:
            session.optimizer.handle_stop()
        if session.ganglion:
            session.ganglion.stop()
        if session.profiler and session.session_dir:
            session.profiler.save_as_csv(Path(session.session_dir))    
    
    logging.info("[Main] All client session cleanups complete.")

    if broadcaster:
        broadcaster.cancel()
        try: await broadcaster
        except asyncio.CancelledError: pass
    
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    
    logging.info("[Main] Server cleanup complete.")

app = FastAPI(lifespan=lifespan)

script_dir = os.path.dirname(__file__)
static_path = os.path.join(script_dir, "front-end", "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
app.mount("/js", StaticFiles(directory="front-end/js"), name="js")

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

async def _heartbeat(websocket: WebSocket, session, stop_event: asyncio.Event):
    '''
    Sends periodic pings to frontend and awaits pong response
    Stops stream immediately if reached timeout with no response
    '''
    hb_settings = CONFIG.get("heartbeat_settings", {})
    
    while not stop_event.is_set():
        
        # Set heartbeat interval (time between pings) based on board status
        status = session.ganglion.connection_status if session.ganglion else "none"
        SERVER_PING_INTERVAL, HEARTBEAT_TIMEOUT = hb_settings.get(status, [10, 30])
        
        await asyncio.sleep(SERVER_PING_INTERVAL)
        
        # Set timeout after sleep state (time to wait for frontend pong)
        new_status = session.ganglion.connection_status if session.ganglion else "none"
        if status == "connected" and new_status != "connected":
            SERVER_PING_INTERVAL, HEARTBEAT_TIMEOUT = hb_settings.get(new_status, [10, 30])
            await asyncio.sleep(SERVER_PING_INTERVAL)
        
        time_since_pong = time.time() - session.last_client_pong
        if time_since_pong > HEARTBEAT_TIMEOUT:
            logging.warning(f"[Main] No client pong response within heartbeat interval")
            await handle_stop_stream(session)
            await manager.disconnect(websocket)
            return
        
        try:
            await websocket.send_json({"type": "server_ping"})
        except Exception as e:
            logging.warning(f"[Main] Failed to send ping: {e}")
            await handle_stop_stream(session)
            await manager.disconnect(websocket)
            return
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    '''
    Handles the lifecycle of a client connection
    Listens for commands to Start/Stop streams specific to this client
    '''
    await manager.connect(websocket)
    session = manager.get_session(websocket)

    hb_stop = asyncio.Event()
    hb_task = asyncio.create_task(_heartbeat(websocket, session, hb_stop))
    
    try:
        while True:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logging.warning("[Main] WebSocket closed by background task. Exiting loop...")
                break
            
            try: # wait for messages 
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
            except asyncio.TimeoutError:
                continue
            
            try: # get command from frontend (stop/start stream or ping/pong)
                command_data = json.loads(data)
                action = command_data.get("action")
                
                if action == "start_calibration":
                    await handle_start_calibration(session, command_data)
                elif action == "generate_gt":
                    await handle_gt_generation(session, command_data)
                elif action == "start_stream":
                    await handle_start_stream(session, command_data)
                elif action == "stop_stream":
                    await handle_stop_stream(session)
                elif action == "client_ping":
                    #logging.info("[SENDING CLIENT PING]")
                    await websocket.send_json({"type": "server_pong"})
                elif action == "client_pong": # received heartbeat response
                    #logging.info("[RECEIVED CLIENT PONG]")
                    session.last_client_pong = time.time()
                    continue
                elif action == "hb_timeout":
                    logging.warning("[Main] Disconnection from client side due to heartbeat timeout")

            except json.JSONDecodeError:
                logging.error("[Main] Received invalid JSON over WebSocket")
    
    except WebSocketDisconnect:
        logging.warning("[Main] WebSocket has disconnected")
    except Exception as e:
        logging.exception(f"[Main] Unexpected error in WebSocket loop: {e}")
        try:
            await websocket.send_json({"status": "error", "message": "Unexpected error in ws loop"})
        except RuntimeError:
            pass
    
    finally:
        hb_stop.set()
        hb_task.cancel()
        await handle_stop_stream(session)
        await manager.disconnect(websocket)

@app.get("/download")
async def download_data(data_path: str, background_tasks: BackgroundTasks):
    ''' 
    Zips the requested folder and serves it as a downloadable file
    '''
    if not data_path or not os.path.exists(data_path):
        return {"error": "Folder not found"}
    
    if os.path.isdir(data_path): # Case 1: data is a folder
        # create a temporary zip archive
        zip_filename = os.path.basename(data_path)
        zip_path = shutil.make_archive(zip_filename, 'zip', data_path)
        
        background_tasks.add_task(os.remove, zip_path) # delete ZIP once done downloading
        
        return FileResponse(zip_path, media_type="application/zip", filename=f"{zip_filename}.zip")
    
    elif os.path.isfile(data_path): # Case 2: data is a single file
        return FileResponse(data_path, filename=os.path.basename(data_path))

async def handle_start_calibration(session, data):
    '''
    Initializes all objects for client calibration (streaming+stimulation) session
    NOTE: This function is only used for calibration sessions
    '''
    if session.ganglion and session.ganglion._emg_thread.is_alive(): 
        await session.websocket.send_json({"status": "error", "message": "Stream already running"})
        return

    folder_name = data.get("folder_name", None)
    serial_port = data.get("serial_port", CONFIG.get('default_serial_port'))
    is_synthetic = data.get("synthetic", CONFIG.get('synthetic'))
    num_trials = data.get("num_trials", CONFIG.get('num_trials'))
    n_reps = data.get("num_reps", CONFIG.get('n_calb_reps'))
    
    # Retrieve reference of current event loop running API server
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = asyncio.get_event_loop()
    
    shared_dorsi_flag = threading.Event()

    # Create session profiler
    session.profiler = SessionProfiler()
    profiler = session.profiler # current session profiler to store timing metrics

    # Initialize Ganglion instance
    def trigger_stream_stop(): # callback function to trigger stream stop in case of board error
        asyncio.run_coroutine_threadsafe(handle_stop_stream(session), current_loop)
    
    if is_synthetic: 
        session.ganglion = SyntheticGanglionData(websocket=session.websocket, profiler=profiler, dorsi_flag=shared_dorsi_flag,
                                                 serial_port=serial_port, sample_rate=250, num_trials=num_trials,
                                                 folder_name=folder_name, on_error=trigger_stream_stop)
    else: 
        session.ganglion = GanglionData(websocket=session.websocket, profiler=profiler, dorsi_flag=shared_dorsi_flag,
                                        serial_port=serial_port, sample_rate=200, num_trials=num_trials,
                                        folder_name=folder_name, on_error=trigger_stream_stop)

    session.session_dir = str(session.ganglion.recorder.session_dir) # current session data folder (initialized within recorder)

    # Initialize Calibrator and Stimulator
    def trigger_auto_stop(): # callback function to trigger auto-stop from Calibrator
        asyncio.run_coroutine_threadsafe(handle_stop_stream(session), current_loop)
    def handle_stim_failed(): # callback function to notify frontend of stimulation failure
        asyncio.run_coroutine_threadsafe(session.websocket.send_json({"type": "stim_failed"}), current_loop)

    session.stimulator = Stimulator(profiler)
    session.calibrator = Calibrator(session.stimulator, profiler, shared_dorsi_flag, n_reps, session.session_dir,
                                    folder_name=folder_name, on_complete=trigger_auto_stop, on_stim_fail=handle_stim_failed)

    # Initialize calibration WatchDog instance
    session.watchdog = WatchDogCalb(profiler, calibrator=session.calibrator)
    directory_to_watch = str(session.ganglion.recorder.features_dir)
    session.watchdog.start_watching(directory_to_watch)
    
    # Start EMG data thread
    session.ganglion.start(current_loop)

    # Attempt to start stream for 20 seconds
    start_time = time.time()
    while time.time() - start_time < 20:
        if session.ganglion.connection_status == "connected":
            await session.websocket.send_json({"status": "success", "message": "EMG streaming started"})
            return
        
        if session.ganglion.connection_status == "failed":
            # error message handled within ganglion object 
            return 

        await asyncio.sleep(0.1)

    # if no successful connection within 20 seconds, assume failure
    await session.websocket.send_json({"status": "error", "message": "Failed to start EMG stream"})

async def handle_gt_generation(session, data):
    '''
    Initializes GTGenerator object and creates ground truth table
    Executes gt generation function in a separate thread to prevent API blocking
    '''
    gt_type = data.get("gt_type", "individual")
    session.gt_generator = GTGenerator(gt_type)

    try:
        loop = asyncio.get_running_loop()
        status, error = await loop.run_in_executor(None, session.gt_generator.run)
    except RuntimeError:
        logging.warning("[Main] Executing gt generation in main FastAPI event loop")
        status = session.gt_generator.run()
    
    if status:
        file_path = Path(__file__).parent / "calibrate" / "ground_truth.csv"
        await session.websocket.send_json({"status": "success", "message": "Ground truth generated", "file": str(file_path)})
    else: 
        await session.websocket.send_json({"status": "error", "type": "GT generation failed", "message": str(error)})

    session.gt_generator = None

async def handle_start_stream(session, data):
    '''
    Initializes all objects for client streaming session
    NOTE: This function is only used for optimization sessions
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
        logging.warning("[Main] No running event loop")
        return
    
    client_id = id(session) # current session client ID for mqtt handling
    shared_dorsi_flag = threading.Event()

    # Create session profiler
    session.profiler = SessionProfiler()
    profiler = session.profiler # current session profiler to store timing metrics
    
    # Initialize Ganglion instance
    def trigger_stream_stop(): # callback function to trigger stream stop in case of board error
        asyncio.run_coroutine_threadsafe(handle_stop_stream(session), current_loop)
    
    if is_synthetic: 
        session.ganglion = SyntheticGanglionData(websocket=session.websocket, profiler=profiler, dorsi_flag=shared_dorsi_flag,
                                                 serial_port=serial_port, sample_rate=250, num_trials=num_trials,
                                                 folder_name=folder_name, on_error=trigger_stream_stop)
    else: 
        session.ganglion = GanglionData(websocket=session.websocket, profiler=profiler, dorsi_flag=shared_dorsi_flag,
                                        serial_port=serial_port, sample_rate=200, num_trials=num_trials,
                                        folder_name=folder_name, on_error=trigger_stream_stop)
    
    session.session_dir = str(session.ganglion.recorder.session_dir) # current session data folder (initialized within recorder)
    
    # Initialize Stimulator and GPBO Optimizer
    def trigger_auto_stop(): # callback function to trigger auto-stop from GPBO
        asyncio.run_coroutine_threadsafe(handle_stop_stream(session), current_loop)
    def handle_stim_failed(): # callback function to notify frontend of stimulation failure
        asyncio.run_coroutine_threadsafe(session.websocket.send_json({"type": "stim_failed"}), current_loop)
    
    session.stimulator = Stimulator(profiler, mqtt_client, client_topic=f"emg/client/{client_id}")
    session.optimizer = GPBOOptimizer(session.stimulator, shared_dorsi_flag, n_iters, n_reps, 
                                      session.session_dir, profiler, mqtt_client, 
                                      client_topic=f"emg/client/{client_id}",
                                      on_complete=trigger_auto_stop, on_stim_fail=handle_stim_failed)
    
    # Initialize optimization WatchDog instance
    session.watchdog = WatchDogOpt(profiler, mqtt_client, client_topic=f"emg/client/{client_id}", 
                                     optimizer=session.optimizer)
    directory_to_watch = str(session.ganglion.recorder.features_dir)
    session.watchdog.start_watching(directory_to_watch)
    
    # Start EMG data thread
    session.ganglion.start(current_loop)

    # Attempt to start stream for 20 seconds
    start_time = time.time()
    while time.time() - start_time < 20:
        if session.ganglion.connection_status == "connected":
            await session.websocket.send_json({"status": "success", "message": "EMG streaming started"})
            return
        
        if session.ganglion.connection_status == "failed":
            # error message handled within ganglion object 
            session.watchdog.stop_watching()
            return 

        await asyncio.sleep(0.1)

    # if no successful connection within 20 seconds, assume failure
    await session.websocket.send_json({"status": "error", "message": "Failed to start EMG stream"})
    session.watchdog.stop_watching()
    
async def handle_stop_stream(session):
    '''
    Stop Ganglion data thread, Watchdog system, Optimizer and Stimulator for client session
    Cleanup of all objects for client session 
    NOTE: This function is used can be used for either optimization sessions or a calibration sessions
    '''
    if session.watchdog:
        session.watchdog.stop_watching()
        session.watchdog = None
    
    if session.optimizer:
        try: 
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, session.optimizer.handle_stop)
        except RuntimeError:
            logging.warning("[Main] Executing optimizer stop in main FastAPI event loop")
            session.optimizer.handle_stop()
        session.optimizer = None

    if session.calibrator:
        try: 
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, session.calibrator.handle_stop)
        except RuntimeError:
            logging.warning("[Main] Executing calibrator stop in main FastAPI event loop")
            session.calibrator.handle_stop()
        if session.calibrator.data_saved:
            try:
                await session.websocket.send_json({"status": "success", "message": "Calibration complete"})
            except (WebSocketDisconnect, RuntimeError):
                pass
        session.calibrator = None

    if session.stimulator:
        session.stimulator = None
    
    if session.ganglion:
        session.ganglion.stop()
        session.ganglion = None
        
        if session.profiler: # export timings once everything has shutdown 
            if session.session_dir:
                try: 
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, session.profiler.save_as_csv, Path(session.session_dir))
                except RuntimeError:
                    logging.warning("[Main] Executing calibrator stop in main FastAPI event loop")
                    session.profiler.save_as_csv(Path(session.session_dir))
                session_dir_path = session.session_dir
                session.session_dir = None
            session.profiler = None
        
        logging.info(f"[Main] Stopped stream/handled cleanup for client {id(session)}")
        try:
            await session.websocket.send_json({"status": "success", "message": "EMG streaming stopped", "folder": session_dir_path})
        except (WebSocketDisconnect, RuntimeError):
            pass
    else:
        try:
            await session.websocket.send_json({"status": "error", "message": "No active stream to stop"})
        except (WebSocketDisconnect, RuntimeError):
            pass

if __name__ == "__main__":
    logging.info("[Main] Starting FastAPI server")
    uvicorn.run(app, host=CONFIG.get('host_IPv4'), port=8000)