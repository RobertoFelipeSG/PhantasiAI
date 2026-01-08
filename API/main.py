import asyncio
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config.connection_manager import manager, logging

from emg.data_acquisition import GanglionData
from emg.synthetic_data_acquisition import SyntheticGanglionData

app = FastAPI()

ganglion_instance = None

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
    global ganglion_instance
    
    if ganglion_instance and ganglion_instance._emg_thread.is_alive(): 
        return {"status": "error", "message": "EMG already streaming"}
    
    # Retrieve reference of event loop running API server
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = asyncio.get_event_loop()
    
    # Create and run EMG execution thread (runs concurrently with FastAPI server)
    if synthetic: 
        ganglion_instance = SyntheticGanglionData(num_trials)
    else: 
        ganglion_instance = GanglionData(num_trials)
    ganglion_instance.start(current_loop)
    
    return {"status": "success", "message": "EMG streaming started"}

@app.post("/stop_stream")
async def stop_stream():
    '''
    Stops CSV recording
    Stops background thread
    '''
    global ganglion_instance
    if ganglion_instance: # Ensure stopping process is complete
        ganglion_instance.stop() 
        ganglion_instance = None
        return {"status": "success", "message": "EMG streaming stopped"}
    return {"status": "error", "message": "No active stream"}

if __name__ == "__main__":
    logging.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)