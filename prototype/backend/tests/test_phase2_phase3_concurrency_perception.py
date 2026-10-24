"""Tests for Phase 2 & 3: Concurrency, Atomic Locking, Perception Error Boundaries, and Depth Engine Integration."""

import concurrent.futures
from datetime import date, datetime, timezone
import io
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import uuid
import numpy as np
from PIL import Image
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.models.base import Base
from app.models.food import Food
from app.models.meal import Meal, MealItem
from app.models.question import DailyBudget
from app.models.triggers import register_i5_triggers
from app.models.user import User
from app.services.allocator.engine import get_budget_remaining, reserve_question_slot
from app.services.vision.depth import DepthVolumeEngine, VolumeEstimate
from app.services.vision.intrinsics import CameraIntrinsics
from app.services.vlm.client import MissingGeminiApiKeyError, VLMClient, VLMItem, VLMOutput
from app.tasks.meal_pipeline import analyze_meal_task, execute_concurrent_dag
from scripts.ingest.seed_nutrition_data import seed_foods


@pytest.fixture
def db_session():
    """Create isolated SQLite in-memory session with I5 triggers and seeded food."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    register_i5_triggers(engine)

    with Session(engine) as session:
        seed_foods(session)
        yield session

    Base.metadata.drop_all(engine)


def test_atomic_reserve_question_slot_concurrency(tmp_path: Path):
    """Verify that reserve_question_slot conditional update strictly prevents over-reservation under concurrency."""
    db_file = tmp_path / "concurrent_test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"timeout": 15},
    )
    Base.metadata.create_all(engine)

    user_id = uuid.uuid4()
    with Session(engine) as session:
        user = User(
            id=user_id,
            email="concurrent_test@messmate.org",
            hashed_password="pw",
            study_arm="A_fixed",
        )
        session.add(user)
        session.commit()

    today = date.today()
    results = []

    def _attempt_reservation():
        with Session(engine) as thread_session:
            success, remaining = reserve_question_slot(
                thread_session,
                user_id=user_id,
                today=today,
                budget_total=3,
            )
            thread_session.commit()
            return success, remaining

    # Concurrently attempt 10 reservations when budget_total is 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_attempt_reservation) for _ in range(10)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    success_count = sum(1 for s, _ in results if s is True)
    failure_count = sum(1 for s, _ in results if s is False)

    assert success_count == 3, f"Expected exactly 3 reservations to succeed, got {success_count}"
    assert failure_count == 7, f"Expected 7 reservations to fail, got {failure_count}"

    with Session(engine) as session:
        budget = session.scalar(
            select(DailyBudget).where(
                DailyBudget.user_id == user_id,
                DailyBudget.local_date == today,
            )
        )
        assert budget is not None
        assert budget.budget_used == 3
        assert budget.budget_total == 3
        assert get_budget_remaining(session, user_id, today) == 0


def test_vlm_live_failure_raises_and_marks_meal_failed(db_session: Session, monkeypatch):
    """Verify runtime Google GenAI failures are never masked with mock foods and mark meals as failed."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")

    vlm = VLMClient(api_key="real-gemini-test-key")

    # Mock _call_gemini_api to simulate a 429 quota exhaustion or network timeout
    def _mock_failing_gemini(*args, **kwargs):
        raise RuntimeError("Google GenAI 429 ResourceExhausted: rate limit exceeded")

    monkeypatch.setattr(vlm, "_call_gemini_api", _mock_failing_gemini)

    # 1. Direct call to extract() must raise RuntimeError rather than returning fake Dal Makhani
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50
    with pytest.raises(RuntimeError) as exc_info:
        vlm.extract(image_bytes=dummy_image)

    assert "Live Gemini API perception failed" in str(exc_info.value)
    assert "429 ResourceExhausted" in str(exc_info.value)

    # 2. Pipeline execution must catch this, fail fast, and persist meal.status = 'failed'
    user = User(
        id=uuid.uuid4(),
        email="vlm_fail_user@messmate.org",
        hashed_password="pw",
    )
    db_session.add(user)
    db_session.commit()

    meal = Meal(
        id=uuid.uuid4(),
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        meal_slot="lunch",
        source="photo",
        status="pending",
        context="personal_home",
        image_key=f"{user.id}/dummy_meal.webp",
    )
    db_session.add(meal)
    db_session.commit()

    # Mock MediaService.get_image_bytes to return image bytes
    from app.services.storage.media import MediaService
    monkeypatch.setattr(MediaService, "get_image_bytes", lambda key: dummy_image)

    # Run analyze_meal_task with the failing live VLMClient
    analyzed = analyze_meal_task(
        db=db_session,
        meal_id=meal.id,
        vlm_client=vlm,
    )

    assert analyzed is not None
    assert analyzed.status == "failed"
    # Ensure no misleading calorie estimates or hallucinated meal items were written to the DB
    items = db_session.scalars(select(MealItem).where(MealItem.meal_id == meal.id)).all()
    assert len(list(items)) == 0


def test_vlm_mock_fallback_allowed_when_appropriate(monkeypatch):
    """Verify mock fallback is cleanly permitted in test environments or when mock_override is passed."""
    # 1. Explicit mock override
    vlm = VLMClient()
    override = VLMOutput(
        plate_type="single_plate",
        items=[VLMItem(dish_label="sambar", vessel_archetype="small_bowl", unit_count=1.0)],
    )
    res, meta = vlm.extract(mock_override=override)
    assert res.items[0].dish_label == "sambar"

    # 2. None image_bytes returns mock fixture without throwing
    res2, meta2 = vlm.extract(image_bytes=None)
    assert len(res2.items) > 0

    # 3. Test environment without TEST_LIVE_API returns deterministic mock
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")
    monkeypatch.delenv("TEST_LIVE_API", raising=False)
    res3, meta3 = vlm.extract(image_bytes=b"some-image-data")
    assert len(res3.items) > 0


def test_depth_volume_engine_pipeline_wiring(db_session: Session):
    """Verify DepthVolumeEngine is wired into the meal pipeline and correctly updates portion volumes."""
    user = User(
        id=uuid.uuid4(),
        email="depth_test@messmate.org",
        hashed_password="pw",
    )
    db_session.add(user)
    db_session.commit()

    meal = Meal(
        id=uuid.uuid4(),
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        meal_slot="dinner",
        source="photo",
        status="pending",
        context="personal_home",
    )
    db_session.add(meal)
    db_session.commit()

    # Synthetic depth map (400x400 at 0.50m)
    h, w = 400, 400
    depth_map = np.ones((h, w), dtype=np.float32) * 0.50

    mock_vlm = VLMOutput(
        plate_type="single_plate",
        items=[
            VLMItem(dish_label="dal makhani", vessel_archetype="standard_bowl", unit_count=1.0),
        ],
    )

    analyzed = analyze_meal_task(
        db=db_session,
        meal_id=meal.id,
        mock_override=mock_vlm,
        depth_map=depth_map,
    )

    assert analyzed is not None
    assert analyzed.status == "analyzed"
    assert analyzed.items[0].portion_source == "model"
    # Depth volume should be computed and non-zero
    assert analyzed.items[0].grams_mean > 0.0
    assert analyzed.items[0].kcal_mean > 0.0
