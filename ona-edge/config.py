"""
ONA Edge Configuration
"""

import os
import logging
from pathlib import Path

# Load .env file if it exists
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Device identification (UUIDs from cloud registration)
DEVICE_ID = os.getenv('ONA_DEVICE_ID', '')
SITE_ID = os.getenv('ONA_SITE_ID', '')
TENANT_ID = os.getenv('ONA_TENANT_ID', '')

# ONA Cloud API
ONA_CLOUD_API = os.getenv('ONA_CLOUD_API', 'https://ona-diagnostic-production.up.railway.app/api/v1')
API_KEY = os.getenv('ONA_API_KEY', '')

# DICOM settings
DICOM_PORT = int(os.getenv('ONA_DICOM_PORT', '11112'))
AE_TITLE = os.getenv('ONA_AE_TITLE', 'ONA_EDGE')

# Storage paths
DATA_DIR = Path(os.getenv('ONA_DATA_DIR', './data'))
STORAGE_DIR = DATA_DIR / 'scans'
MODEL_DIR = DATA_DIR / 'models'

# Ensure directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# AI settings
DEVICE = os.getenv('ONA_DEVICE', 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu')
MAX_WORKERS = int(os.getenv('ONA_MAX_WORKERS', '2'))

# UI settings
UI_PORT = int(os.getenv('ONA_UI_PORT', '8080'))

# Sync settings
SYNC_INTERVAL = int(os.getenv('ONA_SYNC_INTERVAL', '60'))  # seconds
MAX_RETRIES = int(os.getenv('ONA_MAX_RETRIES', '5'))


def setup_logging():
    """Configure logging"""
    log_dir = DATA_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'ona-edge.log')
        ]
    )
