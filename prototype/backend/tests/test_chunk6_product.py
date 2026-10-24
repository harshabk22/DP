"""Integration tests for Chunk 6: Product Web UI, Media Pipeline, and Tableware Calibration."""

import io
from datetime import datetime, timezone
import uuid
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.security import create_access_token
from app.main import app
from app.models.base import Base
from app.models.event import Event
from app.models.meal import Meal
from app.models.prior import UnitPrior
from app.models.triggers import register_i5_triggers
from app.models.user import Consent, Profile, User
from app.services.storage.media import MediaService
from scripts.ingest.seed_nutrition_data import seed_foods


@pytest.fixture
def test_db():
    """Create isolated SQLite test database."""
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


@pytest.fixture
def auth_client(test_db):
    """TestClient with an authenticated test user."""
    user = User(email="chef@messmate.app", hashed_password="hashed_pw_test", study_arm="B_adaptive")
    test_db.add(user)
    test_db.flush()

    profile = Profile(user_id=user.id, birth_year=1990, sex="female", height_cm=165, weight_kg=60)
    consent = Consent(user_id=user.id, purpose="core_logging", granted=True, notice_version="1.0")
    test_db.add_all([profile, consent])
    test_db.commit()

    token = create_access_token(user.id)

    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})

    yield client, user, test_db

    app.dependency_overrides.clear()


def test_static_files_served(auth_client):
    """Verify that root / and static assets are served properly with correct content types."""
    client, _, _ = auth_client

    # Root index.html
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "MessMate" in res_root.text
    assert "Log Meal" in res_root.text or "Snap & Shove" in res_root.text

    # Static CSS
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers.get("content-type", "")
    assert "--canvas-bg" in res_css.text or "--bg-primary" in res_css.text

    # Static JS
    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers.get("content-type", "")
    assert "quickDemoAuth" in res_js.text


def test_tableware_onboarding_seeds_informative_priors(auth_client):
    """Verify Day 1 Visual Kitchen Mirror seeds informative priors (sigma2_0 = 0.05)."""
    client, user, db = auth_client

    payload = {
        "archetypes": [
            {"unit": "standard_bowl", "volume_ml": 350.0, "label": "Standard Bowl"},
            {"unit": "dinner_plate", "volume_ml": 500.0, "label": "Dinner Plate"},
            {"unit": "cup_mug", "volume_ml": 240.0, "label": "Coffee Mug"},
        ]
    }

    res = client.post("/v1/me/onboarding/tableware", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("success", "calibrated")
    assert len(data["seeded_units"]) == 3

    # Verify database state
    priors = db.query(UnitPrior).filter(UnitPrior.user_id == user.id).all()
    assert len(priors) == 3

    bowl_prior = next(p for p in priors if p.unit == "standard_bowl")
    assert bowl_prior.context == "personal_home"
    assert round(bowl_prior.sigma2_log, 3) == 0.050
    assert bowl_prior.n_obs == 1
    # mu_log should equal ln(350)
    import math
    assert abs(bowl_prior.mu_log - math.log(350.0)) < 1e-4

    # Verify audit event logged
    event = db.query(Event).filter(Event.user_id == user.id, Event.type == "tableware_onboarding_completed").first()
    assert event is not None


def test_media_downsampling_and_exif():
    """Verify MediaService creates downsampled WebP (<120KB) and preserves metadata."""
    # Create an oversized in-memory test image (1600x1200 RGB)
    img = Image.new("RGB", (1600, 1200), color=(180, 70, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw_bytes = buf.getvalue()

    user_id = uuid.uuid4()
    meal_id = uuid.uuid4()

    image_key, webp_bytes, exif_meta = MediaService.process_and_save_image(
        image_input=raw_bytes,
        user_id=user_id,
        meal_id=meal_id,
        max_size=800,
        quality=80,
    )

    assert image_key == f"{user_id}/{meal_id}.webp"
    assert len(webp_bytes) < 120_000

    # Verify saved image properties
    saved_img = Image.open(io.BytesIO(webp_bytes))
    assert saved_img.format == "WEBP"
    assert saved_img.width <= 800
    assert saved_img.height <= 800

    # Verify file saved on disk
    file_bytes = MediaService.get_image_bytes(image_key)
    assert file_bytes is not None
    assert len(file_bytes) == len(webp_bytes)


def test_meal_upload_with_image(auth_client, tmp_path):
    """Verify multipart/form-data meal upload processes image and assigns image_key."""
    client, user, db = auth_client

    # Create dummy meal image
    img = Image.new("RGB", (640, 480), color=(80, 150, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/v1/meals",
        data={"context": "personal_home", "meal_slot": "lunch"},
        files={"file": ("plate.jpg", img_bytes, "image/jpeg")},
        params={"sync_run": True},
    )
    assert response.status_code == 202
    data = response.json()
    assert "meal_id" in data
    meal_id = uuid.UUID(data["meal_id"])

    # Query meal
    meal = db.query(Meal).filter(Meal.id == meal_id).first()
    assert meal is not None
    assert meal.image_key is not None
    assert meal.image_purge_at is not None
    assert meal.source == "photo"
