"""Comprehensive test suite for Chunk 1: Database Models and Invariants."""

import uuid
from datetime import date, datetime, timezone
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import is_postgres
from app.models import (
    Base,
    CafeteriaMenu,
    ChatMessage,
    Consent,
    Correction,
    DailyBudget,
    Event,
    Food,
    Meal,
    MealItem,
    ModelRun,
    Profile,
    Question,
    UnitPrior,
    UnitPriorHistory,
    User,
)
from app.models.triggers import (
    hard_delete_session,
    reset_hard_delete_bypass,
    set_hard_delete_bypass,
)


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_config_defaults():
    """Verify application configuration defaults."""
    cfg = Settings()
    assert cfg.DATABASE_URL == "sqlite+aiosqlite:///./messmate.db"
    assert cfg.SYNC_DATABASE_URL == "sqlite:///./messmate.db"
    assert cfg.SECRET_KEY == "messmate-production-secure-hmac-sha256-key-v1-super-secret-512bit"
    assert cfg.JWT_ALGORITHM == "HS256"
    assert cfg.ACCESS_TOKEN_EXPIRE_MINUTES == 1440
    assert cfg.ENVIRONMENT in ("development", "test")


def test_is_postgres_detection(db_session):
    """Verify dialect detection helper on SQLite and PostgreSQL targets."""
    assert is_postgres(db_session) is False
    assert is_postgres(db_session.get_bind()) is False
    assert is_postgres("postgresql+asyncpg://user:pass@localhost/db") is True
    assert is_postgres("sqlite:///./test.db") is False


def test_all_14_models_instantiation(db_session):
    """Instantiate and persist all 14 models, verifying relationships and foreign keys."""
    # 1. User
    user = User(
        email="researcher@messmate.org",
        hashed_password="argon2id$hashedpassword",
        is_active=True,
        study_arm="B_adaptive",
    )
    db_session.add(user)
    db_session.flush()
    assert isinstance(user.id, uuid.UUID)

    # 2. Profile
    profile = Profile(
        user_id=user.id,
        birth_year=1992,
        sex="male",
        height_cm=178.0,
        weight_kg=74.5,
        activity_level="moderate",
        goal="awareness",
        diet_type="eggetarian",
        allergies=["shellfish"],
        conditions=["hypertension"],
        region="South Asia",
        locale="en-IN",
        cafeteria_id="hq_cafeteria",
    )
    db_session.add(profile)

    # 3. Consent
    consent = Consent(
        user_id=user.id,
        purpose="core_logging",
        granted=True,
        notice_version="v2.0",
    )
    db_session.add(consent)

    # 4. Food
    food = Food(
        source="INDB",
        source_code="INDB_3021",
        name_en="Paneer Butter Masala",
        name_local={"hi": "पनीर बटर मसाला"},
        aliases=["paneer makhani"],
        food_group="Dairy & Curries",
        density_g_per_ml=1.08,
        kcal_per_100g=220.0,
        protein_g=9.5,
        fat_g=18.0,
        carb_g=7.5,
        fibre_g=1.2,
        micros={"calcium_mg": 210.0},
        default_unit="standard_bowl",
        pop_grams_per_unit=250.0,
        embedding_json=[0.05, 0.12, -0.33],
        provenance_note="Certified INDB reference row",
    )
    db_session.add(food)
    db_session.flush()
    assert food.id is not None

    # 5. CafeteriaMenu
    menu_item = CafeteriaMenu(
        cafeteria_id="hq_cafeteria",
        day_of_week=2,
        meal_slot="lunch",
        food_id=food.id,
        standard_serving_g=250.0,
    )
    db_session.add(menu_item)

    # 6. Meal
    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date(2026, 9, 21),
        meal_slot="lunch",
        image_key="meals/user1/img_001.webp",
        source="photo",
        status="analyzed",
        context="personal_home",
        kcal_mean=550.0,
        kcal_sd=42.0,
        kcal_p10=495.0,
        kcal_p90=608.0,
        max_ask_value=35.0,
        latency_ms=750,
    )
    db_session.add(meal)
    db_session.flush()

    # 7. MealItem
    meal_item = MealItem(
        meal_id=meal.id,
        food_id=food.id,
        dish_label_raw="bowl of paneer butter masala",
        resolution_tier="exact",
        dish_uncertain=False,
        unit="standard_bowl",
        unit_count=1.0,
        grams_mean=250.0,
        grams_sd=25.0,
        kcal_mean=550.0,
        kcal_sd=42.0,
        ask_value=35.0,
        mu_log=5.52,
        sigma2_log=0.04,
        portion_source="prior",
    )
    db_session.add(meal_item)
    db_session.flush()

    # 8. ModelRun
    model_run = ModelRun(
        meal_id=meal.id,
        model_name="gemini-2.0-flash-lite",
        prompt_version="v1.0.0",
        k_samples=5,
        temperature=0.0,
        raw_responses={"candidates": [{"label": "paneer butter masala", "unit": "standard_bowl"}]},
        latency_ms=640,
        cost_usd=0.00038,
    )
    db_session.add(model_run)

    # 9. Question
    question = Question(
        meal_id=meal.id,
        meal_item_id=meal_item.id,
        question_type="unit_count",
        prompt_text="How much of the bowl did you have?",
        options=["All of it (1 bowl)", "Half bowl (0.5)", "More than 1 (1.5)"],
        ask_value=35.0,
        threshold_used=20.0,
        budget_remaining=2,
        policy_name="messmate_voi",
    )
    db_session.add(question)

    # 10. Correction
    correction = Correction(
        user_id=user.id,
        meal_item_id=meal_item.id,
        field="unit_count",
        old_value={"n": 1.0},
        new_value={"n": 1.0},
        source="question",
    )
    db_session.add(correction)

    # 11. DailyBudget
    budget = DailyBudget(
        user_id=user.id,
        local_date=date(2026, 9, 21),
        budget_total=3,
        budget_used=1,
    )
    db_session.add(budget)

    # 12. UnitPrior
    prior = UnitPrior(
        user_id=user.id,
        unit="standard_bowl",
        context="personal_home",
        mu_log=5.5214,
        sigma2_log=0.0382,
        n_obs=4,
        consecutive_rejects=0,
        last_reject_sign=0,
    )
    db_session.add(prior)

    # 13. UnitPriorHistory
    prior_hist = UnitPriorHistory(
        user_id=user.id,
        unit="standard_bowl",
        context="personal_home",
        mu_log=5.5214,
        sigma2_log=0.0382,
        n_obs=4,
        triggered_by=meal.id,
    )
    db_session.add(prior_hist)

    # 14. Event
    event = Event(
        user_id=user.id,
        type="capture_submitted",
        meta={"source": "camera", "exif": {"focal_length": 4.25}},
    )
    db_session.add(event)

    db_session.commit()

    # Verify relationship navigations
    assert user.profile.birth_year == 1992
    assert len(user.consents) == 1
    assert len(user.meals) == 1
    assert user.meals[0].items[0].dish_label_raw == "bowl of paneer butter masala"
    assert user.meals[0].question.question_type == "unit_count"
    assert user.meals[0].model_runs[0].prompt_version == "v1.0.0"
    assert food.cafeteria_menus[0].cafeteria_id == "hq_cafeteria"
    assert menu_item.food.name_en == "Paneer Butter Masala"


def test_cascade_delete_meals(db_session):
    """Verify cascading deletion from Meal to MealItem, ModelRun, and Question."""
    user = User(email="cascade_user@test.org", hashed_password="pw")
    db_session.add(user)
    db_session.flush()

    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="manual",
        status="pending",
    )
    db_session.add(meal)
    db_session.flush()

    item = MealItem(meal_id=meal.id, dish_label_raw="salad")
    db_session.add(item)
    db_session.flush()

    run = ModelRun(
        meal_id=meal.id,
        model_name="mock-vlm",
        prompt_version="v1",
        k_samples=1,
        raw_responses={},
    )
    db_session.add(run)

    q = Question(
        meal_id=meal.id,
        meal_item_id=item.id,
        question_type="unit_size",
        prompt_text="Size?",
        options=["S", "M"],
        ask_value=10.0,
        budget_remaining=2,
        policy_name="test",
    )
    db_session.add(q)
    db_session.commit()

    # Delete meal without any attached corrections
    db_session.delete(meal)
    db_session.commit()

    assert db_session.scalar(select(Meal).where(Meal.id == meal.id)) is None
    assert db_session.scalar(select(MealItem).where(MealItem.id == item.id)) is None
    assert db_session.scalar(select(ModelRun).where(ModelRun.id == run.id)) is None
    assert db_session.scalar(select(Question).where(Question.id == q.id)) is None


def test_invariant_i3_unique_question_per_meal(db_session):
    """Invariant I3: Max 1 question per meal enforced via database unique constraint."""
    user = User(email="i3_test@messmate.org", hashed_password="pw")
    db_session.add(user)
    db_session.flush()

    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="photo",
        status="analyzed",
    )
    db_session.add(meal)
    db_session.flush()

    q1 = Question(
        meal_id=meal.id,
        question_type="unit_count",
        prompt_text="First question?",
        options=["1", "2"],
        ask_value=15.0,
        budget_remaining=2,
        policy_name="messmate_voi",
    )
    db_session.add(q1)
    db_session.commit()

    # Attempting to insert second question on same meal_id must raise IntegrityError
    q2 = Question(
        meal_id=meal.id,
        question_type="oil_level",
        prompt_text="Second question?",
        options=["low", "medium"],
        ask_value=8.0,
        budget_remaining=1,
        policy_name="messmate_voi",
    )
    db_session.add(q2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_invariant_i5_append_only_corrections(db_session):
    """Invariant I5: `corrections` is strictly append-only; UPDATE & DELETE are blocked."""
    user = User(email="i5_test@messmate.org", hashed_password="pw")
    db_session.add(user)
    db_session.flush()

    corr = Correction(
        user_id=user.id,
        field="grams",
        old_value={"grams": 180},
        new_value={"grams": 220},
        source="scale",
    )
    db_session.add(corr)
    db_session.commit()
    corr_id = corr.id

    # 1. Verify UPDATE is rejected by database trigger
    corr.new_value = {"grams": 250}
    with pytest.raises(IntegrityError) as excinfo:
        db_session.commit()
    assert "Invariant I5 violated" in str(excinfo.value)
    db_session.rollback()

    # 2. Verify DELETE is rejected by database trigger
    c_fetch = db_session.get(Correction, corr_id)
    db_session.delete(c_fetch)
    with pytest.raises(IntegrityError) as excinfo:
        db_session.commit()
    assert "Invariant I5 violated" in str(excinfo.value)
    db_session.rollback()

    # 3. Verify bypass allows hard erasure during complete account deletion
    with hard_delete_session(db_session):
        c_fetch = db_session.get(Correction, corr_id)
        db_session.delete(c_fetch)
        db_session.commit()

    assert db_session.get(Correction, corr_id) is None

    # 4. Verify that once bypass context exits, trigger protection is restored
    corr2 = Correction(
        user_id=user.id,
        field="oil_level",
        old_value={"oil": "medium"},
        new_value={"oil": "low"},
        source="question",
    )
    db_session.add(corr2)
    db_session.commit()

    db_session.delete(corr2)
    with pytest.raises(IntegrityError) as excinfo:
        db_session.commit()
    assert "Invariant I5 violated" in str(excinfo.value)
    db_session.rollback()


def test_invariant_i6_unbundled_consent(db_session):
    """Invariant I6: Consent is unbundled, per-purpose, with independent revocation."""
    user = User(email="i6_test@messmate.org", hashed_password="pw")
    db_session.add(user)
    db_session.flush()

    # Grant 3 distinct purposes
    c_core = Consent(user_id=user.id, purpose="core_logging", granted=True, notice_version="v1.0")
    c_vlm = Consent(user_id=user.id, purpose="model_improvement", granted=True, notice_version="v1.0")
    c_dataset = Consent(user_id=user.id, purpose="research_dataset", granted=False, notice_version="v1.0")
    db_session.add_all([c_core, c_vlm, c_dataset])
    db_session.commit()

    # Verify 3 distinct rows exist
    user_consents = db_session.scalars(select(Consent).where(Consent.user_id == user.id)).all()
    assert len(user_consents) == 3

    # Revoke model_improvement independently
    c_vlm.granted = False
    c_vlm.revoked_at = datetime.now(timezone.utc)
    db_session.commit()

    active_consents = db_session.scalars(
        select(Consent).where(Consent.user_id == user.id, Consent.granted.is_(True))
    ).all()
    assert len(active_consents) == 1
    assert active_consents[0].purpose == "core_logging"


def test_guid_roundtrip_serialization(db_session):
    """Verify GUID TypeDecorator handles Python uuid.UUID and string round-trip."""
    uid = uuid.uuid4()
    user = User(
        id=uid,
        email="guid_test@messmate.org",
        hashed_password="pw",
    )
    db_session.add(user)
    db_session.commit()

    # Query with uuid.UUID object
    q1 = db_session.scalar(select(User).where(User.id == uid))
    assert q1 is not None
    assert isinstance(q1.id, uuid.UUID)
    assert q1.id == uid

    # Query with string representation
    q2 = db_session.scalar(select(User).where(User.id == str(uid)))
    assert q2 is not None
    assert q2.id == uid


def test_jsonb_and_array_compat_roundtrip(db_session):
    """Verify JSONB_compat and Array_compat serialize and deserialize correctly."""
    user = User(email="compat_test@messmate.org", hashed_password="pw")
    db_session.add(user)
    db_session.flush()

    profile = Profile(
        user_id=user.id,
        birth_year=1998,
        sex="other",
        allergies=["gluten", "tree_nuts", "soy"],
        conditions=["celiac", "pcos"],
    )
    db_session.add(profile)

    food = Food(
        source="CURATED",
        source_code="CUR_99",
        name_en="Multi-grain Khichdi",
        name_local={"hi": "खिचड़ी", "gu": "ખીચડી"},
        aliases=["khichdi", "khichuri"],
        kcal_per_100g=135.0,
        micros={"sodium_mg": 280, "potassium_mg": 140},
        embedding_json=[0.11, 0.22, 0.33],
    )
    db_session.add(food)
    db_session.commit()

    q_prof = db_session.scalar(select(Profile).where(Profile.user_id == user.id))
    assert q_prof.allergies == ["gluten", "tree_nuts", "soy"]
    assert q_prof.conditions == ["celiac", "pcos"]

    q_food = db_session.scalar(select(Food).where(Food.source_code == "CUR_99"))
    assert q_food.name_local == {"hi": "खिचड़ी", "gu": "ખીચડી"}
    assert q_food.aliases == ["khichdi", "khichuri"]
    assert q_food.micros["sodium_mg"] == 280
    assert q_food.embedding == [0.11, 0.22, 0.33]


def test_sync_db_dependency():
    """Verify get_db dependency yields active sync session."""
    from sqlalchemy import text
    from app.core.database import get_db

    gen = get_db()
    session = next(gen)
    try:
        val = session.execute(text("SELECT 1")).scalar()
        assert val == 1
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_async_db_dependency():
    """Verify get_async_db dependency yields active async session."""
    from sqlalchemy import text
    from app.core.database import get_async_db

    agen = get_async_db()
    session = await anext(agen)
    try:
        val = (await session.execute(text("SELECT 1"))).scalar()
        assert val == 1
    finally:
        await agen.aclose()


def test_chat_message_and_meal_item_extensions(db_session):
    """Verify ChatMessage model creation, user relationship, and MealItem schema extensions."""
    from app.schemas.meal import MealItemDetail, MealItemUpdate

    # 1. User & ChatMessage creation
    user = User(
        email="chat_test@messmate.org",
        hashed_password="pw_hash",
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    msg_user = ChatMessage(
        user_id=user.id,
        role="user",
        message="Did I have enough protein today?",
    )
    msg_assistant = ChatMessage(
        user_id=user.id,
        role="assistant",
        message="You have consumed 42g of protein so far. A dinner with dal or paneer will hit your 54g target.",
    )
    db_session.add_all([msg_user, msg_assistant])
    db_session.commit()

    # Query chat messages
    messages = db_session.scalars(
        select(ChatMessage).where(ChatMessage.user_id == user.id).order_by(ChatMessage.created_at)
    ).all()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].message == "Did I have enough protein today?"
    assert isinstance(messages[0].id, uuid.UUID)
    assert messages[0].created_at is not None
    assert messages[0].user.email == "chat_test@messmate.org"
    assert len(user.chat_messages) == 2

    # 2. MealItem vlm_grams_raw and is_refill validation
    meal = Meal(
        user_id=user.id,
        captured_at=datetime.now(timezone.utc),
        local_date=date.today(),
        source="photo",
        status="analyzed",
    )
    db_session.add(meal)
    db_session.flush()

    item_standard = MealItem(
        meal_id=meal.id,
        dish_label_raw="dal tadka",
        grams_mean=180.0,
    )
    item_refill = MealItem(
        meal_id=meal.id,
        dish_label_raw="roti",
        vlm_grams_raw=45.0,
        grams_mean=40.0,
        is_refill=True,
    )
    db_session.add_all([item_standard, item_refill])
    db_session.commit()

    # Verify standard item defaults
    db_session.refresh(item_standard)
    assert item_standard.is_refill is False
    assert item_standard.vlm_grams_raw is None

    # Verify refill item attributes
    db_session.refresh(item_refill)
    assert item_refill.is_refill is True
    assert item_refill.vlm_grams_raw == 45.0

    # 3. Pydantic schema validation
    detail_refill = MealItemDetail(
        id=item_refill.id,
        dish_label=item_refill.dish_label_raw,
        grams_mean=item_refill.grams_mean,
        is_refill=item_refill.is_refill,
        vlm_grams_raw=item_refill.vlm_grams_raw,
    )
    assert detail_refill.is_refill is True
    assert detail_refill.vlm_grams_raw == 45.0

    detail_standard = MealItemDetail(
        id=item_standard.id,
        dish_label=item_standard.dish_label_raw,
    )
    assert detail_standard.is_refill is False
    assert detail_standard.vlm_grams_raw is None

    # Test MealItemUpdate schema with is_refill
    update_payload = MealItemUpdate(is_refill=True, grams=60.0)
    assert update_payload.is_refill is True
    assert update_payload.grams == 60.0

    # 4. Verify cascade delete of chat_messages when user deleted
    user_id = user.id
    db_session.delete(user)
    db_session.commit()

    remaining_messages = db_session.scalars(
        select(ChatMessage).where(ChatMessage.user_id == user_id)
    ).all()
    assert len(remaining_messages) == 0

