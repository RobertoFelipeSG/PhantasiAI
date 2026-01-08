import logging
import asyncio
import json 
from typing import List, Optional
from fastapi import WebSocket

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True 
)

# ----- WebSocket Manager: Handles browser connections ----- # 
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None  # Store the loop here

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info("New WebSocket connected")

        if self.main_loop is None:
            try:
                self.main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logging.info("WebSocket disconnected")

    # Broadcast to connected clients 
    async def broadcast(self, message: str):
        if not self.active_connections:
            return
        
        try: # Use gather for concurrent sending to all clients
            await asyncio.gather(*[
                conn.send_text(message) 
                for conn in self.active_connections
            ], return_exceptions=True)
        except Exception as e:
            print(f"Error broadcasting message: {e}")

manager = ConnectionManager()

class WebSocketLogHandler(logging.Handler):
    """
    Send Python logs to the Frontend via WebSocket
    """
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
ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(ws_handler)