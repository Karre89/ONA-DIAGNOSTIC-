from fastapi import APIRouter

from app.api.devices import router as devices_router
from app.api.results import router as results_router
from app.api.feedback import router as feedback_router
from app.api.models_api import router as models_router
from app.api.seed import router as seed_router
from app.api.auth import router as auth_router
from app.api.referrals import router as referrals_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(devices_router)
api_router.include_router(results_router)
api_router.include_router(feedback_router)
api_router.include_router(models_router)
api_router.include_router(seed_router)
api_router.include_router(referrals_router)
