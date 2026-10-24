"""All MessMate database models and declarative base."""

from app.models.base import Array_compat, Base, BigInt_compat, GUID, JSONB_compat
from app.models.user import Consent, Profile, User
from app.models.food import CafeteriaMenu, Food
from app.models.meal import Meal, MealItem, ModelRun
from app.models.question import Correction, DailyBudget, Question
from app.models.prior import UnitPrior, UnitPriorHistory
from app.models.event import Event
from app.models.chat import ChatMessage
from app.models.triggers import register_i5_triggers

__all__ = [
    "Base",
    "GUID",
    "JSONB_compat",
    "Array_compat",
    "BigInt_compat",
    "User",
    "Profile",
    "Consent",
    "Food",
    "CafeteriaMenu",
    "Meal",
    "MealItem",
    "ModelRun",
    "Question",
    "Correction",
    "DailyBudget",
    "UnitPrior",
    "UnitPriorHistory",
    "Event",
    "ChatMessage",
    "register_i5_triggers",
]
