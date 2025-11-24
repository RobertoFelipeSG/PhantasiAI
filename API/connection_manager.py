import logging
import asyncio
from typing import List
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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info("New WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logging.info("WebSocket disconnected")

    # Broadcast to connected clients 
    async def broadcast(self, message: str):
        await asyncio.gather(*[ # use gather for concurrent sending
            conn.send_text(message) 
            for conn in self.active_connections
        ], return_exceptions=True)

manager = ConnectionManager()