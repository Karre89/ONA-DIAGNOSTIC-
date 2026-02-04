"""
ONA Edge DICOM Listener
Receives X-ray images from clinic equipment
"""

import logging
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import Verification

from .handlers import handle_store, handle_echo
import sys
sys.path.append('..')
from config import DICOM_PORT, AE_TITLE

logger = logging.getLogger('ona.dicom')


class DICOMListener:
    """DICOM Service Class Provider (SCP) for receiving images"""

    def __init__(self, port: int = DICOM_PORT, ae_title: str = AE_TITLE):
        self.port = port
        self.ae_title = ae_title
        self.ae = None
        self._running = False
        self._inference_callback = None

    def set_inference_callback(self, callback):
        """Set callback function for when images are received"""
        self._inference_callback = callback

    def setup(self):
        """Initialize Application Entity"""
        self.ae = AE(ae_title=self.ae_title)

        # Accept all storage SOP classes
        self.ae.supported_contexts = AllStoragePresentationContexts

        # Also support verification (C-ECHO)
        self.ae.add_supported_context(Verification)

        # Set timeouts
        self.ae.acse_timeout = 30
        self.ae.dimse_timeout = 30
        self.ae.network_timeout = 30

        logger.info(f"DICOM AE '{self.ae_title}' configured")

    def start(self):
        """Start listening for incoming associations"""
        if not self.ae:
            self.setup()

        # Create handlers with access to inference callback
        def store_handler(event):
            return handle_store(event, self._inference_callback)

        handlers = [
            (evt.EVT_C_STORE, store_handler),
            (evt.EVT_C_ECHO, handle_echo),
            (evt.EVT_CONN_OPEN, self._on_connect),
            (evt.EVT_CONN_CLOSE, self._on_disconnect),
        ]

        logger.info(f"Starting DICOM listener on port {self.port}")
        self._running = True

        # Blocking call - runs until stopped
        self.ae.start_server(
            ('0.0.0.0', self.port),
            evt_handlers=handlers,
            block=True
        )

    def stop(self):
        """Stop the listener"""
        self._running = False
        if self.ae:
            self.ae.shutdown()
            logger.info("DICOM listener stopped")

    def _on_connect(self, event):
        """Log new connections"""
        logger.info(f"DICOM connection from {event.address}")

    def _on_disconnect(self, event):
        """Log disconnections"""
        logger.debug(f"DICOM disconnected: {event.address}")


# Singleton instance
_listener = None


def get_listener() -> DICOMListener:
    global _listener
    if _listener is None:
        _listener = DICOMListener()
    return _listener
