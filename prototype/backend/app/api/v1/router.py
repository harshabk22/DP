"""Aggregated API v1 Router."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.consents import router as consents_router
from app.api.v1.health import router as health_router
from app.api.v1.meal_items import router as meal_items_router
from app.api.v1.meals import router as meals_router
from app.api.v1.questions import router as questions_router
from app.api.v1.rollups import router as rollups_router
from app.api.v1.users import router as users_router

# Unprefixed API router aggregating all endpoints
api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(consents_router)
api_router.include_router(meals_router)
api_router.include_router(questions_router)
api_router.include_router(meal_items_router)
api_router.include_router(rollups_router)
api_router.include_router(health_router)
api_router.include_router(chat_router)

# Canonical /v1 router
api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(api_router)


