import logging
import asyncio
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import DeviceState

logger = logging.getLogger(__name__)


class DeviceDaemon:
    """
    Device Daemon - manages device lifecycle

    Features:
    - Registers device with cloud on first start
    - Sends periodic heartbeats
    - Checks for model updates
    - Reports health metrics
    """

    def __init__(self, db: Session):
        self.db = db
        self.cloud_url = settings.cloud_api_url

    def get_device_state(self) -> Optional[DeviceState]:
        """Get current device registration state"""
        return self.db.query(DeviceState).filter(DeviceState.id == "default").first()

    def save_device_state(
        self,
        device_id: str,
        tenant_id: str,
        site_id: str,
        device_token: str
    ):
        """Save device registration state"""
        state = self.get_device_state()
        if state:
            state.device_id = device_id
            state.tenant_id = tenant_id
            state.site_id = site_id
            state.device_token = device_token
            state.registered_at = datetime.utcnow()
        else:
            state = DeviceState(
                id="default",
                device_id=device_id,
                tenant_id=tenant_id,
                site_id=site_id,
                device_token=device_token,
                registered_at=datetime.utcnow()
            )
            self.db.add(state)

        self.db.commit()
        logger.info(f"Device state saved: device_id={device_id}")

    async def register_device(self, tenant_id: str, site_code: str, device_name: str) -> bool:
        """Register device with cloud"""
        logger.info(f"Registering device: {device_name}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.cloud_url}/api/v1/devices/register",
                    json={
                        "tenant_id": tenant_id,
                        "site_code": site_code,
                        "device_name": device_name,
                        "hardware_info": self._get_hardware_info()
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    self.save_device_state(
                        device_id=data["device_id"],
                        tenant_id=tenant_id,
                        site_id=data["site_id"],
                        device_token=data["device_token"]
                    )
                    logger.info(f"Device registered: {data['device_id']}")
                    return True
                else:
                    logger.error(f"Registration failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    async def send_heartbeat(self) -> bool:
        """Send heartbeat to cloud"""
        state = self.get_device_state()
        if not state or not state.device_id:
            logger.warning("Device not registered, skipping heartbeat")
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.cloud_url}/api/v1/devices/{state.device_id}/heartbeat",
                    json={
                        "timestamp": datetime.utcnow().isoformat(),
                        "edge_version": "0.1.0",
                        "models_loaded": [settings.default_model_version],
                        "health": self._get_health_metrics()
                    },
                    headers={"Authorization": f"Bearer {state.device_token}"}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Process any commands from cloud
                    if data.get("commands"):
                        await self._process_commands(data["commands"])
                    return True
                else:
                    logger.warning(f"Heartbeat failed: {response.status_code}")
                    return False

        except httpx.ConnectError:
            logger.debug("Cloud unreachable for heartbeat")
            return False
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return False

    async def check_model_updates(self) -> list:
        """Check cloud for model updates"""
        state = self.get_device_state()
        if not state or not state.device_id:
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.cloud_url}/api/v1/models/manifest",
                    params={"device_id": state.device_id},
                    headers={"Authorization": f"Bearer {state.device_token}"}
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])

        except Exception as e:
            logger.error(f"Model check error: {e}")

        return []

    async def _process_commands(self, commands: list):
        """Process commands from cloud"""
        for cmd in commands:
            logger.info(f"Processing command: {cmd}")
            # TODO: Implement command handlers

    def _get_hardware_info(self) -> dict:
        """Get hardware information"""
        return {
            "platform": "linux",
            "cpu": "unknown",
            "memory_gb": 16,
            "disk_gb": 256,
            "gpu": None
        }

    def _get_health_metrics(self) -> dict:
        """Get current health metrics"""
        return {
            "cpu_percent": 25.0,
            "memory_percent": 45.0,
            "disk_percent": 30.0,
            "studies_today": 0,
            "errors_today": 0
        }


async def run_heartbeat_loop(db_session_factory):
    """Background task to send heartbeats"""
    logger.info("Starting heartbeat loop")

    while True:
        try:
            db = db_session_factory()
            try:
                daemon = DeviceDaemon(db)
                await daemon.send_heartbeat()
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")

        await asyncio.sleep(30)  # Heartbeat every 30 seconds
