import logging
import asyncio
import json
import time
import csv
from datetime import datetime
from typing import List, Optional
from fastapi import WebSocket
from pathlib import Path

log_path = Path(__file__).parent / "testing_logs.txt"

logging.basicConfig(
    #filename=log_path,
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True 
)

# ----- WebSocket Manager: Handles browser connections ----- # 
class SessionProfiler:
    def __init__(self):
        self.trials = {}
        self.current_trial = 0

    def start_trial(self, trial_num):
        '''Initialize new trial dictionary for current trial'''
        self.current_trial = trial_num
        self.trials[trial_num] =  {
            "overall_process": None,
            "file_create": None,
            "feat_extract": None,
            "opt_iter": None,
            "stim": None,
            "overall_opt_stim": None,
            "_trial_end_time": time.time() 
        }

    def log_metric(self, trial_num, metric_name, duration):
        """Saves a specific timing metric to trial dictionary"""
        if trial_num in self.trials:
            self.trials[trial_num][metric_name] = duration

    def mark_process_complete(self, trial_num):
        """
        Store overall time from when a trial ended to when stimulation completed
        (One entire iteration process)
        """
        if trial_num in self.trials:
            start_time = self.trials[trial_num]["_trial_end_time"]
            self.trials[trial_num]["overall_process"] = time.time() - start_time

    def save_as_csv(self, base_path: Path):
        """Converts dict to CSV file at end of the session"""
        if not self.trials:
            return # Nothing to export
            
        filepath = base_path / "timings.csv"
        
        # Define the exact columns you asked for
        fieldnames = [
            "trial_num",
            "overall_process",
            "file_create",
            "feat_extract",
            "opt_iter",
            "stim",
            "overall_opt_stim",
        ]

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for t_num, metrics in self.trials.items():
                    row = {"trial_num": t_num}
                    for field in fieldnames[1:]: 
                        row[field] = metrics.get(field, "ERROR") # if no time, process unsuccesful
                    writer.writerow(row)
        except OSError as e:
            logging.info(f"[Profiler] Could not save session timing metrics: {e}")

class ClientSession:
    def __init__(self, websocket: WebSocket):
        '''Initialize single client connection'''
        self.websocket = websocket
        self.session_dir = None
        self.ganglion = None
        self.optimizer = None
        self.stimulator = None
        self.watchdog_feat = None
        self.profiler = SessionProfiler()

    async def cleanup(self):
        '''Clean up resources when client disconnects'''
        if self.watchdog_feat:
            self.watchdog_feat.stop_watching()
            self.watchdog_feat = None
        if self.stimulator:
            self.stimulator = None
        if self.optimizer:
            self.optimizer = None
        if self.session_dir:
            self.session_dir = None
        if self.ganglion:
            self.ganglion.stop()
            self.ganglion = None
        if self.profiler:
            self.profiler = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket, ClientSession] = {} # Map WebSocket to ClientSession
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None  # Store the loop here

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = ClientSession(websocket)
        logging.info(f"[Manager] New Client connected. Total connections: {len(self.active_connections)}")

        if self.main_loop is None:
            try:
                self.main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            await self.active_connections[websocket].cleanup()
            del self.active_connections[websocket]
            logging.info("[Manager] Client disconnected.")

    def get_session(self, websocket: WebSocket) -> Optional[ClientSession]:
        return self.active_connections.get(websocket)
    
    async def broadcast(self, message: str):
        '''Broadcast to ALL connected clients'''
        if not self.active_connections:
            return
        
        try: # Use gather for concurrent sending to all clients
            await asyncio.gather(*[
                ws.send_text(message) 
                for ws in self.active_connections.keys()
            ], return_exceptions=True)
        except Exception as e:
            print(f"Error broadcasting message: {e}")

manager = ConnectionManager()

# ----- Send Python logs to the Frontend via WebSocket ----- #
class WebSocketLogHandler(logging.Handler):
    def __init__(self, ws_manager):
        super().__init__()
        self.manager = ws_manager

    def emit(self, record):
        try:
            log_entry = self.format(record)
            
            payload = json.dumps({
                "type": "server_log", 
                "message": log_entry
            })
            
            try:
                # 1: Schedule on the CURRENT loop (main thread)
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.broadcast(payload))
            except RuntimeError:
                # 2: If we are in a background thread, use the stored main_loop
                if self.manager.main_loop and self.manager.main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.manager.broadcast(payload), 
                        self.manager.main_loop
                    )
                
        except Exception:
            self.handleError(record)

ws_handler = WebSocketLogHandler(manager)
ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
logging.getLogger().addHandler(ws_handler)