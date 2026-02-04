"""
ONA Edge Sync Service
Handles communication with ONA Cloud
"""

import asyncio
import logging
import httpx
import hashlib
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import json
import sqlite3

import sys
sys.path.append('..')
from config import (
    ONA_CLOUD_API,
    API_KEY,
    DEVICE_ID,
    SITE_ID,
    SYNC_INTERVAL,
    MAX_RETRIES,
    DATA_DIR
)

logger = logging.getLogger('ona.sync')


class SyncService:
    """Handles synchronization with ONA Cloud"""

    def __init__(self, db_path: str = None):
        self.api_base = ONA_CLOUD_API
        self.api_key = API_KEY
        self.db_path = db_path or str(DATA_DIR / 'sync_queue.db')
        self._online = False
        self._last_sync = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite queue database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT UNIQUE,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                attempts INTEGER DEFAULT 0,
                last_attempt TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')
        conn.commit()
        conn.close()

    async def check_connectivity(self) -> bool:
        """Check if ONA Cloud is reachable"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.api_base.replace('/api/v1', '')}/health",
                    headers=self._get_headers()
                )
                self._online = response.status_code == 200
        except Exception as e:
            logger.debug(f"Connectivity check failed: {e}")
            self._online = False
        return self._online

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers"""
        headers = {
            'Content-Type': 'application/json',
            'X-Device-ID': DEVICE_ID
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    async def queue_result(self, result: Dict):
        """Add scan result to sync queue"""
        scan_id = result.get('scan_id')

        # Prepare payload (de-identified, no images)
        payload = {
            'tenant_id': SITE_ID,  # Will be mapped on server
            'device_id': DEVICE_ID,
            'study_id': scan_id,
            'modality': 'CXR',
            'model_version': 'tb_v1.0',
            'inference_time_ms': result.get('processing_time_ms'),
            'scores': {},
            'risk_bucket': 'LOW',
            'input_hash': hashlib.sha256(scan_id.encode()).hexdigest()
        }

        # Add condition results
        for condition, data in result.get('conditions', {}).items():
            payload['scores'][condition] = data.get('probability', 0)
            payload['risk_bucket'] = data.get('severity', 'LOW')

        # Insert into queue
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                'INSERT OR REPLACE INTO sync_queue (scan_id, payload) VALUES (?, ?)',
                (scan_id, json.dumps(payload))
            )
            conn.commit()
            logger.debug(f"Queued scan for sync: {scan_id}")
        finally:
            conn.close()

        # Try immediate sync if online
        if self._online:
            await self.sync_pending()

    async def sync_pending(self):
        """Sync all pending results to cloud"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get pending items
        cursor.execute('''
            SELECT id, scan_id, payload, attempts
            FROM sync_queue
            WHERE status = 'pending' AND attempts < ?
            ORDER BY created_at
            LIMIT 50
        ''', (MAX_RETRIES,))

        pending = cursor.fetchall()
        conn.close()

        if not pending:
            return

        logger.info(f"Syncing {len(pending)} pending results")

        async with httpx.AsyncClient(timeout=30.0) as client:
            for row_id, scan_id, payload_json, attempts in pending:
                try:
                    payload = json.loads(payload_json)

                    response = await client.post(
                        f"{self.api_base}/results",
                        json=payload,
                        headers=self._get_headers()
                    )

                    if response.status_code in (200, 201):
                        self._mark_synced(row_id)
                        logger.debug(f"Synced: {scan_id}")
                    else:
                        self._mark_failed(row_id, attempts + 1)
                        logger.warning(f"Sync failed for {scan_id}: {response.status_code}")

                except Exception as e:
                    self._mark_failed(row_id, attempts + 1)
                    logger.error(f"Sync error for {scan_id}: {e}")

        self._last_sync = datetime.utcnow()

    def _mark_synced(self, row_id: int):
        """Mark item as successfully synced"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE sync_queue SET status = 'synced' WHERE id = ?",
            (row_id,)
        )
        conn.commit()
        conn.close()

    def _mark_failed(self, row_id: int, attempts: int):
        """Mark sync attempt as failed"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE sync_queue SET attempts = ?, last_attempt = ? WHERE id = ?",
            (attempts, datetime.utcnow().isoformat(), row_id)
        )
        conn.commit()
        conn.close()

    async def send_heartbeat(self):
        """Send heartbeat to cloud"""
        if not self._online:
            return

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'edge_version': '1.0.0',
                    'models_loaded': ['tb_v1.0'],
                    'health': {
                        'cpu_percent': 50,
                        'memory_percent': 60,
                        'disk_percent': 40
                    }
                }

                await client.post(
                    f"{self.api_base}/devices/{DEVICE_ID}/heartbeat",
                    json=payload,
                    headers=self._get_headers()
                )
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")

    async def check_model_updates(self) -> Optional[dict]:
        """Check for model updates from cloud"""
        if not self._online:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base}/models/manifest",
                    params={'device_id': DEVICE_ID},
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.debug(f"Model update check failed: {e}")

        return None

    async def run_sync_loop(self):
        """Main sync loop - runs continuously"""
        logger.info("Starting sync service")
        while True:
            try:
                await self.check_connectivity()

                if self._online:
                    await self.sync_pending()
                    await self.send_heartbeat()

                # Update UI sync status
                from ui.server import set_sync_status
                set_sync_status(self._online, self._last_sync)

            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            await asyncio.sleep(SYNC_INTERVAL)

    @property
    def is_online(self) -> bool:
        return self._online

    @property
    def last_sync(self) -> Optional[datetime]:
        return self._last_sync


# Singleton
_sync_service = None


def get_sync_service() -> SyncService:
    global _sync_service
    if _sync_service is None:
        _sync_service = SyncService()
    return _sync_service
