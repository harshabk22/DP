"""Conversational Nutrition Co-Pilot endpoint for MessMate."""

from datetime import datetime, timezone
import re
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.safety import get_user_daily_targets
from app.models.chat import ChatMessage
from app.models.meal import Meal
from app.models.user import Profile, User
from app.schemas.chat import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    MacroOverview,
)
from app.services.digest.engine import WeeklyDigestEngine

router = APIRouter(prefix="/chat", tags=["assistant"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with the Nutrition Co-Pilot (Grounded Guidance)",
)
def chat_with_copilot(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Conversational assistant grounded in user's actual daily nutrition logs."""
    target_date = payload.date_query or datetime.now(timezone.utc).date()

    # Query today's logged meals
    meals = (
        db.query(Meal)
        .filter(
            Meal.user_id == current_user.id,
            Meal.local_date == target_date,
            Meal.status != "discarded",
        )
        .all()
    )

    consumed_kcal = 0.0
    protein_g = 0.0
    fat_g = 0.0
    carb_g = 0.0

    for m in meals:
        if m.kcal_mean:
            consumed_kcal += float(m.kcal_mean)
        for item in m.items:
            if item.food and item.grams_mean:
                ratio = float(item.grams_mean) / 100.0
                protein_g += float(item.food.protein_g or 0.0) * ratio
                fat_g += float(item.food.fat_g or 0.0) * ratio
                carb_g += float(item.food.carb_g or 0.0) * ratio

    profile = (
        getattr(current_user, "profile", None)
        or db.query(Profile).filter(Profile.user_id == current_user.id).first()
    )
    target_kcal, target_protein, target_carb, target_fat = get_user_daily_targets(profile)

    consumed_kcal = round(consumed_kcal, 1)
    remaining_kcal = max(0.0, round(target_kcal - consumed_kcal, 1))
    protein_g = round(protein_g, 1)
    carb_g = round(carb_g, 1)
    fat_g = round(fat_g, 1)

    macros = MacroOverview(
        target_kcal=target_kcal,
        consumed_kcal=consumed_kcal,
        remaining_kcal=remaining_kcal,
        protein_g=protein_g,
        carb_g=carb_g,
        fat_g=fat_g,
    )

    # Generate grounded response
    msg_lower = payload.message.lower()

    if any(k in msg_lower for k in ["protein", "how much protein", "protein left"]):
        remaining_p = max(0.0, round(target_protein - protein_g, 1))
        reply = (
            f"You've had {protein_g}g of protein today out of your {target_protein}g goal "
            f"({remaining_p}g remaining). You still have {remaining_kcal} kcal left in your daily budget!"
        )
    elif any(k in msg_lower for k in ["dinner", "what to eat", "suggest", "order", "food swap", "recommend"]):
        engine = WeeklyDigestEngine(db)
        advice, swaps = engine.generate_dinner_recommendation(current_user.id, target_date)
        reply = advice
    elif any(k in msg_lower for k in ["how am i doing", "status", "summary", "calories left", "calories"]):
        reply = (
            f"Today you've logged {consumed_kcal} kcal ({remaining_kcal} kcal remaining to your {target_kcal} kcal target). "
            f"Macros so far: {protein_g}g protein, {carb_g}g carbs, {fat_g}g fat. You're doing great!"
        )
    else:
        reply = (
            f"I'm here to help you hit your nutrition goals! Today you've logged {consumed_kcal} kcal "
            f"with {remaining_kcal} kcal remaining. Feel free to ask me for dinner suggestions or how your macros are tracking."
        )

    # Persist conversation turns to ChatMessage entity (Chunk 5)
    user_turn = ChatMessage(
        user_id=current_user.id,
        role="user",
        message=payload.message,
    )
    bot_turn = ChatMessage(
        user_id=current_user.id,
        role="assistant",
        message=reply,
    )
    db.add_all([user_turn, bot_turn])
    db.commit()

    return ChatResponse(reply=reply, macros=macros)


@router.get(
    "/history",
    response_model=list[ChatMessageResponse],
    summary="Get recent conversation history",
)
def get_chat_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatMessageResponse]:
    """Fetch chronologically ordered chat history for current user."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return [ChatMessageResponse.model_validate(m) for m in messages]
