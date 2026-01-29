import logging
import asyncio
import json 
from typing import List, Optional
from fastapi import WebSocket

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True 
)

# ----- WebSocket Manager: Handles browser connections ----- # 
class ClientSession:
    def __init__(self, websocket: WebSocket):
        '''Initialize single client connection'''
        self.websocket = websocket
        self.ganglion = None
        self.watchdog_class = None

    async def cleanup(self):
        '''Clean up resources when client disconnects'''
        if self.watchdog_class:
            self.watchdog_class.stop_watching()
        if self.ganglion:
            self.ganglion.stop()

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