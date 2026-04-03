import logging
import asyncio
import json
import time
import queue
from datetime import datetime
from typing import List, Optional
from fastapi import WebSocket
from pathlib import Path

log_path = Path(__file__).parent / "testing_logs.txt"

logging.basicConfig(
    #filename=log_path,
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    force=True 
)

# ----- WebSocket Manager: Handles browser connections ----- # 
class ClientSession:
    def __init__(self, websocket: WebSocket):
        '''Initialize single client connection'''
        self.websocket = websocket
        self.last_client_pong = time.time()
        self.session_dir = None
        self.ganglion = None
        self.optimizer = None
        self.stimulator = None
        self.watchdog_feat = None
        self.profiler = None

    async def cleanup(self):
        '''
        Clean up resources when client disconnects
        '''
        if self.watchdog_feat:
            logging.warning("[Manager] Watchdog found. Cleaning up resource.")
            self.watchdog_feat.stop_watching()
            self.watchdog_feat = None
        
        if self.stimulator:
            logging.warning("[Manager] Stimulator found. Cleaning up resource")
            self.stimulator = None
        
        if self.optimizer:
            logging.warning("[Manager] Optimizer found. Cleaning up resource")
            # create optimization data backup incase of sudden disconnection
            self.optimizer.handle_stop()
            self.optimizer = None
        
        if self.ganglion:
            logging.warning("[Manager] Ganglion found. Cleaning up resource")
            self.ganglion.stop()
            self.ganglion = None
        
        if self.profiler:
            logging.warning("[Manager] Profiler found. Cleaning up resource")
            # create timing metrics data backup incase of sudden disconnection
            if self.session_dir: 
                self.profiler.save_as_csv(Path(self.session_dir))
            self.profiler = None
        
        if self.session_dir:
            self.session_dir = None

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

            try:
                await websocket.close(code=1000)
                logging.info(f"[Manager] Websocket closed succesfully")
            except Exception as e:
                logging.error(f"[Manager] Error closing websocket manually: {e}")

            logging.info(f"[Manager] Client disconnected. Total connections: {len(self.active_connections)}")

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
        self.log_queue = queue.Queue()

    def emit(self, record):
        try:
            log_entry = self.format(record)
            payload = json.dumps({"type": "server_log", "message": log_entry})
            self.log_queue.put(payload)
        except Exception:
            self.handleError(record)

ws_handler = WebSocketLogHandler(manager)
ws_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', '%H:%M:%S'))
logging.getLogger().addHandler(ws_handler)