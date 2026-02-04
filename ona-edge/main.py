"""
ONA Edge - Main Application
AI-Powered Medical Diagnostics at the Edge

This is the main entry point that orchestrates all components:
- DICOM Listener (receives X-rays from machines)
- AI Inference Engine (runs TB detection)
- Local UI (nurse-facing web interface)
- Sync Service (cloud communication)
"""

import asyncio
import logging
import signal
import sys
import threading
from pathlib import Path

from config import setup_logging, DEVICE_ID, DICOM_PORT, UI_PORT
from dicom import get_listener
from inference import get_engine
from sync import get_sync_service
from ui.server import run_ui_server, add_result

logger = logging.getLogger('ona.main')


def process_scan(scan_id: str, filepath: str, metadata: dict):
    """
    Process a received scan through the AI pipeline.
    Called by DICOM handler when an image is received.
    """
    logger.info(f"Processing scan: {scan_id}")

    # Run AI inference
    engine = get_engine()
    result = engine.run_inference(scan_id, filepath, conditions=['tb'])

    if result['success']:
        # Add to UI for display
        add_result(result)

        # Queue for cloud sync
        sync_service = get_sync_service()
        asyncio.run(sync_service.queue_result(result))

        logger.info(f"Scan complete: {scan_id} | TB: {result['conditions'].get('tb', {}).get('probability', 0)}%")
    else:
        logger.error(f"Scan failed: {scan_id} | Error: {result.get('error')}")


async def run_dicom_listener():
    """Run DICOM listener in async context"""
    listener = get_listener()
    listener.set_inference_callback(process_scan)

    # Run in thread pool (blocking call)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, listener.start)


async def main():
    """Main application entry point"""
    setup_logging()

    print("=" * 60)
    print(f"  ONA Edge - AI-Powered Medical Diagnostics")
    print(f"  Device: {DEVICE_ID}")
    print("=" * 60)

    logger.info("Starting ONA Edge...")

    # Initialize AI engine
    logger.info("Loading AI models...")
    engine = get_engine()
    engine.initialize()

    # Initialize sync service
    logger.info("Initializing sync service...")
    sync_service = get_sync_service()

    # Start UI server in separate thread
    logger.info(f"Starting UI server on port {UI_PORT}...")
    ui_thread = threading.Thread(target=run_ui_server, daemon=True)
    ui_thread.start()

    # Start services
    logger.info(f"Starting DICOM listener on port {DICOM_PORT}...")

    print()
    print(f"  DICOM: Listening on port {DICOM_PORT}")
    print(f"  UI: http://localhost:{UI_PORT}")
    print()
    print("  Ready to receive X-rays!")
    print("=" * 60)

    # Run async services
    tasks = [
        asyncio.create_task(run_dicom_listener()),
        asyncio.create_task(sync_service.run_sync_loop()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down...")


def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received shutdown signal, exiting...")
    sys.exit(0)


if __name__ == '__main__':
    # Handle shutdown signals
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run main loop
    asyncio.run(main())
