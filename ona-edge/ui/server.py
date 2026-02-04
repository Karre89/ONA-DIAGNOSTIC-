"""
ONA Edge Local UI Server
Serves the web interface for healthcare workers
"""

import asyncio
import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import sys
sys.path.append('..')
from config import UI_PORT, DEVICE_ID

app = FastAPI(title="ONA Edge UI")

# Paths
ui_dir = Path(__file__).parent
static_path = ui_dir / "static"
templates_path = ui_dir / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# Store for current results (in-memory)
current_results: Dict[str, dict] = {}
connected_clients: List[WebSocket] = []

# Sync status
sync_status = {
    'online': False,
    'last_sync': None
}


def set_sync_status(online: bool, last_sync: datetime = None):
    """Update sync status"""
    global sync_status
    sync_status['online'] = online
    if last_sync:
        sync_status['last_sync'] = last_sync.isoformat()


def add_result(result: dict):
    """Add new scan result and notify clients"""
    scan_id = result.get('scan_id')
    if scan_id:
        # Convert numpy arrays to base64 for JSON
        result_copy = result.copy()
        if 'conditions' in result_copy:
            for condition, data in result_copy['conditions'].items():
                if 'heatmap' in data and isinstance(data['heatmap'], np.ndarray):
                    # Encode as base64 PNG
                    _, buffer = cv2.imencode('.png', data['heatmap'])
                    data['heatmap_b64'] = base64.b64encode(buffer).decode('utf-8')
                    del data['heatmap']
                if 'overlay' in data and isinstance(data['overlay'], np.ndarray):
                    _, buffer = cv2.imencode('.png', data['overlay'])
                    data['overlay_b64'] = base64.b64encode(buffer).decode('utf-8')
                    del data['overlay']
        if 'original_image' in result_copy:
            if isinstance(result_copy['original_image'], np.ndarray):
                _, buffer = cv2.imencode('.png', result_copy['original_image'])
                result_copy['original_image_b64'] = base64.b64encode(buffer).decode('utf-8')
                del result_copy['original_image']

        current_results[scan_id] = result_copy

        # Notify WebSocket clients
        asyncio.create_task(broadcast_result(result_copy))


async def broadcast_result(result: dict):
    """Send result to all connected WebSocket clients"""
    for client in connected_clients:
        try:
            await client.send_json({"type": "new_scan", "data": result})
        except:
            pass


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "device_id": DEVICE_ID,
        "sync_status": sync_status
    })


@app.get("/scan/{scan_id}", response_class=HTMLResponse)
async def view_scan(request: Request, scan_id: str):
    """View result for specific scan"""
    result = current_results.get(scan_id)
    if not result:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "Scan not found"
        })
    return templates.TemplateResponse("scan_result.html", {
        "request": request,
        "result": result,
        "device_id": DEVICE_ID
    })


@app.get("/api/scans")
async def list_scans(limit: int = 50):
    """API: List recent scans"""
    scans = list(current_results.values())
    scans.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return JSONResponse(scans[:limit])


@app.get("/api/scans/{scan_id}")
async def get_scan(scan_id: str):
    """API: Get specific scan result"""
    result = current_results.get(scan_id)
    if not result:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(result)


@app.get("/api/status")
async def get_status():
    """API: Get device status"""
    return JSONResponse({
        "device_id": DEVICE_ID,
        "engine_ready": True,
        "sync_status": sync_status,
        "total_scans": len(current_results)
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time scan notifications"""
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Send heartbeat every 30 seconds
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


def run_ui_server():
    """Start the UI server"""
    uvicorn.run(app, host="0.0.0.0", port=UI_PORT, log_level="info")
