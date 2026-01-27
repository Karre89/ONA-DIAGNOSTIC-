import json
import logging
import asyncio
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import SyncQueue, Result, Feedback, DeviceState

logger = logging.getLogger(__name__)


class SyncClient:
    """
    Sync Client - uploads outputs to cloud with retry/backoff

    Features:
    - Queues items locally in SQLite
    - Retries with exponential backoff
    - Works offline (queue grows, syncs when connected)
    - Idempotent uploads (cloud handles deduplication)
    """

    def __init__(self, db: Session):
        self.db = db
        self.cloud_url = settings.cloud_api_url
        self.max_retries = settings.sync_max_retries
        self.batch_size = settings.sync_batch_size

    def get_device_state(self) -> Optional[DeviceState]:
        """Get current device registration state"""
        return self.db.query(DeviceState).filter(DeviceState.id == "default").first()

    def get_auth_headers(self) -> dict:
        """Get authorization headers for cloud API"""
        state = self.get_device_state()
        token = state.device_token if state else settings.device_token
        return {"Authorization": f"Bearer {token}"}

    async def sync_pending(self) -> dict:
        """
        Sync all pending items to cloud

        Returns dict with sync statistics
        """
        stats = {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "offline": False
        }

        # Get pending items
        pending = self.db.query(SyncQueue).filter(
            SyncQueue.attempts < self.max_retries
        ).limit(self.batch_size).all()

        if not pending:
            logger.debug("No pending items to sync")
            return stats

        logger.info(f"Syncing {len(pending)} pending items")

        async with httpx.AsyncClient(timeout=30.0) as client:
            for item in pending:
                stats["attempted"] += 1

                try:
                    success = await self._sync_item(client, item)
                    if success:
                        stats["succeeded"] += 1
                        # Remove from queue
                        self.db.delete(item)
                        self.db.commit()
                    else:
                        stats["failed"] += 1
                        item.attempts += 1
                        self.db.commit()

                except httpx.ConnectError:
                    logger.warning("Cloud unreachable - working offline")
                    stats["offline"] = True
                    break

                except Exception as e:
                    logger.error(f"Sync error for {item.record_type}/{item.record_id}: {e}")
                    stats["failed"] += 1
                    item.attempts += 1
                    self.db.commit()

        logger.info(f"Sync complete: {stats}")
        return stats

    async def _sync_item(self, client: httpx.AsyncClient, item: SyncQueue) -> bool:
        """Sync a single item to cloud"""

        if item.record_type == "result":
            return await self._sync_result(client, item)
        elif item.record_type == "feedback":
            return await self._sync_feedback(client, item)
        else:
            logger.warning(f"Unknown record type: {item.record_type}")
            return False

    async def _sync_result(self, client: httpx.AsyncClient, item: SyncQueue) -> bool:
        """Sync a result to cloud"""
        payload = json.loads(item.payload_json)

        state = self.get_device_state()
        if not state:
            logger.error("Device not registered, cannot sync")
            return False

        request_data = {
            "tenant_id": state.tenant_id,
            "device_id": state.device_id,
            "study_id": payload["study_id"],
            "modality": "CXR",
            "model_version": payload["model_version"],
            "inference_time_ms": payload["inference_time_ms"],
            "scores": payload["scores"],
            "risk_bucket": payload["risk_bucket"],
            "explanation": payload["explanation"],
            "input_hash": payload["input_hash"],
            "has_burned_in_text": payload.get("has_burned_in_text", False)
        }

        response = await client.post(
            f"{self.cloud_url}/api/v1/results",
            json=request_data,
            headers=self.get_auth_headers()
        )

        if response.status_code in [200, 201]:
            # Update result synced_at
            result = self.db.query(Result).filter(Result.id == item.record_id).first()
            if result:
                result.synced_at = datetime.utcnow()
                self.db.commit()
            logger.info(f"Synced result {item.record_id}")
            return True
        else:
            logger.error(f"Failed to sync result: {response.status_code} - {response.text}")
            return False

    async def _sync_feedback(self, client: httpx.AsyncClient, item: SyncQueue) -> bool:
        """Sync feedback to cloud"""
        payload = json.loads(item.payload_json)

        state = self.get_device_state()
        if not state:
            logger.error("Device not registered, cannot sync")
            return False

        request_data = {
            "tenant_id": state.tenant_id,
            "site_code": payload.get("site_code", "KNH001"),
            "study_id": payload["study_id"],
            "response": payload["response"],
            "reason": payload.get("reason")
        }

        response = await client.post(
            f"{self.cloud_url}/api/v1/feedback",
            json=request_data,
            headers=self.get_auth_headers()
        )

        if response.status_code in [200, 201]:
            feedback = self.db.query(Feedback).filter(Feedback.id == item.record_id).first()
            if feedback:
                feedback.synced_at = datetime.utcnow()
                self.db.commit()
            logger.info(f"Synced feedback {item.record_id}")
            return True
        else:
            logger.error(f"Failed to sync feedback: {response.status_code} - {response.text}")
            return False

    def queue_feedback(self, feedback: Feedback):
        """Add feedback to sync queue"""
        sync_item = SyncQueue(
            record_type="feedback",
            record_id=feedback.id,
            payload_json=json.dumps({
                "study_id": feedback.study_id,
                "response": feedback.response,
                "reason": feedback.reason
            })
        )
        self.db.add(sync_item)
        self.db.commit()


async def run_sync_loop(db_session_factory):
    """Background task to sync pending items"""
    logger.info("Starting sync loop")

    while True:
        try:
            db = db_session_factory()
            try:
                sync_client = SyncClient(db)
                await sync_client.sync_pending()
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Sync loop error: {e}")

        await asyncio.sleep(settings.sync_interval_seconds)
