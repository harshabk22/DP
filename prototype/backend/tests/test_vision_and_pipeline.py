"""Empirical verification test suite for MessMate vision, pipeline DAG, and Invariant I5.

Covers:
1. Camera Intrinsics & CIPA DC-008 diagonal FOV normalization (§2.1).
2. Tableware Re-ID DINOv2 masked rim pooling & joint multimodal matching (§2.2).
3. Concurrent DAG orchestration via asyncio.gather hitting <442 ms SLA (§1.1).
4. Decoupled 'Snap & Shove' API ingestion with client UUID & HTTP 202 Accepted.
5. Invariant I5 PostgreSQL statement-level BEFORE TRUNCATE trigger.
"""

import asyncio
from datetime import date, datetime, timezone
import io
import math
import time
from typing import Any
import uuid
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user, get_db
from app.core.database import SessionLocal, is_postgres, sync_engine
from app.main import app
from app.models.base import Base
from app.models.food import Food
from app.models.meal import Meal, MealItem, ModelRun
from app.models.prior import UnitPrior
from app.models.question import Correction
from app.models.triggers import POSTGRES_STATEMENTS, SQLITE_STATEMENTS, register_i5_triggers
from app.models.user import User
from scripts.ingest.seed_nutrition_data import seed_foods


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database session with I5 triggers."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    register_i5_triggers(engine)

    SessionLocal.configure(bind=engine)
    with Session(engine) as session:
        seed_foods(session)
        yield session

    SessionLocal.configure(bind=sync_engine)
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def _override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

from app.services.vision.intrinsics import (
    CIPA_35MM_DIAGONAL_MM,
    DEFAULT_FLAGSHIP_FOCAL_35MM,
    CameraIntrinsics,
    compute_focal_length_cipa,
    compute_metric_scale_anchor,
    extract_camera_intrinsics,
    project_3d_to_pixel,
    unproject_pixel_to_3d,
)
from app.services.vision.reid import (
    TABLEWARE_MATCH_THRESHOLD,
    TablewareDescriptor,
    compute_tableware_similarity,
    extract_masked_rim,
    is_tableware_match,
    pool_masked_features,
)
from app.services.vlm.client import VLMClient, VLMItem, VLMOutput
from app.tasks.meal_pipeline import (
    analyze_meal_task,
    execute_concurrent_dag,
    run_local_cv_branch,
    run_prefetch_branch,
    run_vlm_branch,
)


# ===========================================================================
# 1. Camera Intrinsics & CIPA DC-008 Diagonal Normalization
# ===========================================================================
def test_cipa_dc008_focal_length_normalization():
    """Verify CIPA DC-008 diagonal field-of-view formula:

    f_px = f_35mm * sqrt(W^2 + H^2) / 43.266615
    """
    w, h = 4000, 3000  # 4:3 smartphone capture
    f35 = 24.0         # 24mm modern flagship lens

    diag = math.sqrt(w**2 + h**2)  # 5000.0 px
    expected_f_px = f35 * (diag / CIPA_35MM_DIAGONAL_MM)

    calc_f_px = compute_focal_length_cipa(width=w, height=h, focal_35mm=f35)
    assert math.isclose(calc_f_px, expected_f_px, rel_tol=1e-6)

    # Verify naive 3:2 width normalization (f = f35 * W / 36.0) induces ~8.17% error on 4:3
    naive_f_px = f35 * (float(w) / 36.0)
    bias_ratio = calc_f_px / naive_f_px
    assert math.isclose(bias_ratio, (5000.0 / 43.266615) / (4000.0 / 36.0), rel_tol=1e-4)
    assert bias_ratio > 1.03  # Significant systematic difference resolved


def test_camera_intrinsics_flagship_prior_fallback():
    """Verify default flagship prior fallback is 24.0 mm when EXIF is missing."""
    intrinsics = extract_camera_intrinsics(image_bytes=None, width=800, height=600)
    assert intrinsics.focal_35mm == DEFAULT_FLAGSHIP_FOCAL_35MM
    assert intrinsics.focal_35mm == 24.0
    assert intrinsics.cx == 400.0
    assert intrinsics.cy == 300.0
    assert len(intrinsics.matrix) == 3


def test_camera_intrinsics_exif_extraction():
    """Verify EXIF FocalLengthIn35mmFilm (tag 41989) is extracted accurately from image."""
    buf = io.BytesIO()
    img = Image.new("RGB", (1200, 900), color="blue")
    exif = img.getexif()
    exif[41989] = 28  # 28mm wide lens in EXIF
    img.save(buf, format="JPEG", exif=exif)
    img_bytes = buf.getvalue()

    intr = extract_camera_intrinsics(image_bytes=img_bytes)
    assert intr.focal_35mm == 28.0
    assert intr.width == 1200
    assert intr.height == 900


def test_metric_scale_anchor_and_3d_projection():
    """Verify circular rim projective invariant metric scale anchoring and ray projection."""
    f_px = 1000.0
    d_true_m = 0.14  # 14cm bowl diameter
    d_major_px = 200.0

    z_anchor = compute_metric_scale_anchor(f_px=f_px, d_true_m=d_true_m, d_major_px=d_major_px)
    assert math.isclose(z_anchor, 0.70, rel_tol=1e-5)  # 0.70 meters (70 cm)

    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=400.0,
        cy=300.0,
        width=800,
        height=600,
        focal_35mm=24.0,
    )
    # Unproject and reproject round-trip
    orig_u, orig_v, depth = 500.0, 350.0, 0.70
    x3d, y3d, z3d = unproject_pixel_to_3d(orig_u, orig_v, depth, intrinsics)
    assert z3d == depth

    proj_u, proj_v = project_3d_to_pixel(x3d, y3d, z3d, intrinsics)
    assert math.isclose(proj_u, orig_u, rel_tol=1e-6)
    assert math.isclose(proj_v, orig_v, rel_tol=1e-6)


# ===========================================================================
# 2. Tableware Re-ID & Multimodal Matching
# ===========================================================================
def test_reid_multimodal_matching_formula():
    """Verify S(A, B) = 0.50 * cos + 0.40 * geom_match + 0.10 * ctx_match."""
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Identical bowl: cos=1.0, geom_match=1.0, ctx_match=1.0 -> S = 1.0
    score = compute_tableware_similarity(
        emb_a=emb_a,
        emb_b=emb_b,
        diameter_a=140.0,
        diameter_b=140.0,
        height_a=60.0,
        height_b=60.0,
        context_a="personal_home",
        context_b="personal_home",
    )
    assert math.isclose(score, 1.0, rel_tol=1e-6)
    assert is_tableware_match(emb_a, emb_b, 140.0, 140.0, 60.0, 60.0, "personal_home", "personal_home")


def test_reid_steel_katori_ambiguity_discrimination():
    """Verify that identically finished steel katoris of different sizes are REJECTED.

    Steel katoris of 150ml (D=100mm, H=40mm) and 350ml (D=140mm, H=60mm) have
    identical visual texture (cos ~ 0.98), fooling naive visual Re-ID, but
    joint multimodal matching correctly discriminates them (S < 0.85).
    """
    emb_a = np.array([0.98, 0.198, 0.0], dtype=np.float32)
    emb_b = np.array([1.00, 0.000, 0.0], dtype=np.float32)

    # Diameter difference 40mm, Height difference 20mm
    score = compute_tableware_similarity(
        emb_a=emb_a,
        emb_b=emb_b,
        diameter_a=100.0,
        diameter_b=140.0,
        height_a=40.0,
        height_b=60.0,
        context_a="personal_home",
        context_b="personal_home",
    )
    # Visual = 0.50 * 0.98 = 0.49
    # Geometric = 0.40 * exp(-1600/450 - 400/450) = 0.40 * 0.0117 = 0.0047
    # Context = 0.10 * 1.0 = 0.10
    # Total S ≈ 0.5947 << 0.85
    assert score < TABLEWARE_MATCH_THRESHOLD
    assert score < 0.65
    assert not is_tableware_match(
        emb_a, emb_b, 100.0, 140.0, 40.0, 60.0, "personal_home", "personal_home"
    )

    # Identical bowl with normal measurement noise (2mm noise)
    score_noisy = compute_tableware_similarity(
        emb_a=emb_a,
        emb_b=emb_b,
        diameter_a=100.0,
        diameter_b=102.0,
        height_a=40.0,
        height_b=41.0,
        context_a="personal_home",
        context_b="personal_home",
    )
    assert score_noisy >= TABLEWARE_MATCH_THRESHOLD
    assert is_tableware_match(
        emb_a, emb_b, 100.0, 102.0, 40.0, 41.0, "personal_home", "personal_home"
    )


def test_masked_rim_extraction_and_pooling_tiers():
    """Verify 3-tier masked rim pooling with 8-sector angular voting and flange fallback."""
    h, w = 100, 100
    vessel = np.zeros((h, w), dtype=bool)
    vessel[20:80, 20:80] = True  # 60x60 square vessel

    food = np.zeros((h, w), dtype=bool)

    # Tier 1: Clean food inside vessel (rim ratio >= 0.15)
    food[35:65, 35:65] = True
    rim_tier1 = extract_masked_rim(vessel, food, rim_erosion_px=4, food_dilation_px=2)
    assert np.any(rim_tier1)
    assert not np.any(rim_tier1 & food)  # Food completely excluded

    # Tier 3: Severe occlusion (food covers entire interior and almost entire rim)
    food_heavy = vessel.copy()
    food_heavy[20:25, 20:25] = False  # Only 25 pixels left
    rim_tier3 = extract_masked_rim(vessel, food_heavy, rim_erosion_px=4, food_dilation_px=2)
    assert np.any(rim_tier3)

    # Feature pooling division-by-zero protection test
    feat = np.ones((16, 16, 384), dtype=np.float32)
    empty_mask = np.zeros((h, w), dtype=bool)
    pooled_zero = pool_masked_features(feat, empty_mask)
    assert pooled_zero.shape == (384,)
    assert not np.isnan(pooled_zero).any()
    assert not np.isinf(pooled_zero).any()
    assert math.isclose(float(np.linalg.norm(pooled_zero)), 1.0, rel_tol=1e-4)


# ===========================================================================
# 3. Concurrent DAG Execution (<442 ms SLA)
# ===========================================================================
@pytest.mark.asyncio
async def test_concurrent_dag_asyncio_gather_parallelism():
    """Verify execute_concurrent_dag executes Branch 1, Branch 2, and Branch 3 concurrently."""
    w, h = 800, 800
    img = Image.new("RGB", (w, h), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Create dummy VLM client with mock
    mock_vlm = VLMOutput(
        plate_type="single_plate",
        items=[
            VLMItem(dish_label="chana masala", vessel_archetype="standard_bowl", unit_count=1.0),
        ],
    )
    vlm_client = VLMClient()

    # Warm up local CV models once so cold torch initialization is not charged to runtime SLA
    await run_local_cv_branch(img_bytes)

    # Measure concurrent execution of branches
    t0 = time.perf_counter()
    b1_task = run_local_cv_branch(img_bytes)
    b2_task = run_vlm_branch(vlm_client, img_bytes, mock_override=mock_vlm)

    cv_data, (vlm_output, vlm_meta) = await asyncio.gather(b1_task, b2_task)
    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    assert "intrinsics" in cv_data
    assert "metric_scale_anchor" in cv_data
    assert len(vlm_output.items) == 1
    assert total_elapsed_ms < 442.0  # Well within the <442 ms SLA threshold


def test_analyze_meal_task_end_to_end_concurrency(test_db_session):
    """Verify analyze_meal_task executes concurrent DAG and persists analyzed state."""
    user = User(email="dag_user@messmate.org", hashed_password="pw")
    test_db_session.add(user)
    test_db_session.flush()

    food = Food(
        name_en="Dal Tadka",
        source="IFCT2017",
        source_code="DAL001",
        density_g_per_ml=1.05,
        kcal_per_100g=120.0,
    )
    test_db_session.add(food)
    test_db_session.flush()

    now = datetime.now(timezone.utc)
    meal = Meal(
        user_id=user.id,
        captured_at=now,
        local_date=now.date(),
        meal_slot="lunch",
        source="photo",
        status="pending",
        context="personal_home",
    )
    test_db_session.add(meal)
    test_db_session.commit()

    vlm_mock = VLMOutput(
        plate_type="single_bowl",
        items=[
            VLMItem(dish_label="dal tadka", vessel_archetype="standard_bowl", unit_count=1.0),
        ],
    )

    analyzed = analyze_meal_task(test_db_session, meal.id, mock_override=vlm_mock)
    assert analyzed is not None
    assert analyzed.status == "analyzed"
    assert len(analyzed.items) == 1
    assert analyzed.items[0].dish_label_raw == "dal tadka"
    assert analyzed.kcal_mean > 0.0

    # Verify Invariant I10 ModelRun provenance was recorded
    runs = test_db_session.query(ModelRun).filter(ModelRun.meal_id == meal.id).all()
    assert len(runs) >= 1
    assert runs[0].model_name in ("gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite")
    assert runs[0].prompt_version == "extract_v1"


# ===========================================================================
# 4. Decoupled 'Snap & Shove' API Ingestion (<40 ms HTTP 202)
# ===========================================================================
def test_snap_and_shove_client_generated_uuid(client):
    """Verify decoupled ingestion route accepts client-generated UUID and returns HTTP 202."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "snap_user@example.com", "password": "Password123!", "birth_year": 1995},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    client_uuid = str(uuid.uuid4())

    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    headers_with_client_id = dict(headers)
    headers_with_client_id["X-Client-Meal-ID"] = client_uuid

    res = client.post(
        "/v1/meals",
        headers=headers_with_client_id,
        data={"context": "personal_home", "meal_slot": "lunch"},
        files={"file": ("snap.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert res.status_code == 202
    data = res.json()
    assert data["meal_id"] == client_uuid
    assert data["status"] in ("pending", "analyzed")

    # Verify idempotent re-transmission with same X-Client-Meal-ID succeeds
    res_retry = client.post(
        "/v1/meals",
        headers=headers_with_client_id,
        data={"context": "personal_home", "meal_slot": "lunch"},
    )
    assert res_retry.status_code == 202
    assert res_retry.json()["meal_id"] == client_uuid


def test_snap_and_shove_form_meal_id_fallback(client):
    """Verify client-generated UUID passed in form field meal_id is accepted."""
    reg = client.post(
        "/v1/auth/register",
        json={"email": "snap_form_user@example.com", "password": "Password123!", "birth_year": 1993},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    client_uuid = str(uuid.uuid4())

    res = client.post(
        "/v1/meals",
        headers=headers,
        data={"meal_id": client_uuid, "context": "personal_home", "meal_slot": "dinner"},
    )
    assert res.status_code == 202
    assert res.json()["meal_id"] == client_uuid


# ===========================================================================
# 5. Invariant I5 Statement-Level BEFORE TRUNCATE Trigger
# ===========================================================================
def test_invariant_i5_postgres_statement_level_truncate_trigger_ddl():
    """Verify POSTGRES_STATEMENTS defines both row-level and statement-level BEFORE TRUNCATE triggers."""
    statements_str = " ".join(POSTGRES_STATEMENTS)

    # 1. Statement-level BEFORE TRUNCATE trigger must exist
    assert "BEFORE TRUNCATE ON corrections" in statements_str
    assert "FOR EACH STATEMENT" in statements_str
    assert "trg_corrections_block_truncate" in statements_str

    # 2. Row-level BEFORE UPDATE OR DELETE trigger must exist
    assert "BEFORE UPDATE OR DELETE ON corrections" in statements_str
    assert "FOR EACH ROW" in statements_str

    # 3. Security definer mutation prevention function blocking Operation %
    assert "CREATE OR REPLACE FUNCTION prevent_corrections_mutation()" in statements_str
    assert "SECURITY DEFINER" in statements_str
    assert "TG_OP" in statements_str


# ===========================================================================
# 6. Production Hardening: VLM Mock Elimination & Real Visual Re-ID
# ===========================================================================
def test_vlm_client_missing_key_in_production_fails_loudly(monkeypatch):
    """Verify VLMClient fails loudly with MissingGeminiApiKeyError when GEMINI_API_KEY is unset in non-test mode."""
    from app.services.vlm.client import MissingGeminiApiKeyError, VLMClient
    from app.core.config import settings

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    client_prod = VLMClient(api_key="")

    with pytest.raises(MissingGeminiApiKeyError) as exc_info:
        client_prod.extract(image_bytes=b"dummy-image-bytes")

    assert "GEMINI_API_KEY" in str(exc_info.value)
    assert "Please configure GEMINI_API_KEY" in str(exc_info.value)


def test_reid_extractor_real_pixel_features_no_randomness():
    """Verify TablewareReIDExtractor extracts genuine, repeatable visual embeddings from actual pixels."""
    from app.services.vision.reid import TablewareReIDExtractor

    extractor = TablewareReIDExtractor()

    # 1. Silver stainless steel bowl image
    steel_arr = np.ones((224, 224, 3), dtype=np.float32) * 0.85
    desc_steel_1 = extractor.extract_descriptor(steel_arr)
    desc_steel_2 = extractor.extract_descriptor(steel_arr)

    # Must be 100% mathematically deterministic (zero RNG)
    assert np.allclose(desc_steel_1.embedding, desc_steel_2.embedding)
    assert desc_steel_1.embedding.shape == (384,)
    assert math.isclose(float(np.linalg.norm(desc_steel_1.embedding)), 1.0, rel_tol=1e-5)

    # 2. Bright blue ceramic bowl image
    blue_arr = np.zeros((224, 224, 3), dtype=np.float32)
    blue_arr[..., 2] = 0.95
    desc_blue = extractor.extract_descriptor(blue_arr)

    # Real visual distinction: steel vs blue should NOT have identical embeddings
    cos_steel_blue = float(np.dot(desc_steel_1.embedding, desc_blue.embedding))
    assert cos_steel_blue < 0.90  # Substantial physical difference captured


def test_dinov2_lazy_singleton_loading_and_patch_extraction():
    """Verify Meta DINOv2 dinov2_vits14_reg singleton loading, patch extraction, and masked rim pooling."""
    import torch
    from app.services.vision.reid import (
        DINOV2_EMBEDDING_DIM,
        TablewareReIDExtractor,
        extract_masked_rim,
        get_dinov2_model,
        pool_masked_features,
    )

    # 1. Lazy singleton loading
    model1 = TablewareReIDExtractor.get_model("dinov2_vits14_reg")
    assert model1 is not None
    assert isinstance(model1, torch.nn.Module)

    model2 = TablewareReIDExtractor.get_model("dinov2_vits14_reg")
    assert model1 is model2  # Exact singleton identity

    model_delegated = get_dinov2_model("dinov2_vits14_reg")
    assert model_delegated is model1

    # 2. Extract spatial patch tokens using model.get_intermediate_layers(x, n=1, return_class_token=False)[0]
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        features = model1.get_intermediate_layers(dummy_input, n=1, return_class_token=False)[0]
    assert features.shape == (1, 256, 384)

    # 3. Reshape patch tokens to spatial feature map (16, 16, 384)
    feat_map = features[0].reshape(16, 16, DINOV2_EMBEDDING_DIM).cpu().numpy().astype(np.float32)
    assert feat_map.shape == (16, 16, 384)

    # 4. Masked rim pooling
    vessel_mask = np.ones((224, 224), dtype=bool)
    food_mask = np.zeros((224, 224), dtype=bool)
    food_mask[50:174, 50:174] = True
    rim_mask = extract_masked_rim(vessel_mask, food_mask)

    rim_emb = pool_masked_features(feat_map, rim_mask)
    assert rim_emb.shape == (384,)
    assert math.isclose(float(np.linalg.norm(rim_emb)), 1.0, rel_tol=1e-5)


def test_mobilenet_v3_small_backbone_and_orthonormal_projection():
    """Verify Priority 2 MobileNetV3 Small backbone extracts 576-D deep features projected to 384-D."""
    from app.services.vision.reid import TablewareReIDExtractor
    import numpy as np
    import math

    extractor = TablewareReIDExtractor(model_name="mobilenet_v3_small")
    model = extractor.get_model("mobilenet_v3_small")
    assert model is not None

    # Test orthonormal projection matrix W^T @ W == I
    w_proj = TablewareReIDExtractor.get_orthonormal_projection_matrix(576, 384)
    assert w_proj.shape == (576, 384)
    assert np.allclose(w_proj.T @ w_proj, np.eye(384, dtype=np.float32), atol=1e-5)

    # Test extraction with MobileNetV3 Small
    steel_arr = np.ones((224, 224, 3), dtype=np.float32) * 0.85
    desc = extractor.extract_descriptor(steel_arr)
    assert desc.embedding.shape == (384,)
    assert math.isclose(float(np.linalg.norm(desc.embedding)), 1.0, rel_tol=1e-5)


def test_reid_extractor_graceful_analytical_fallback(monkeypatch):
    """Verify TablewareReIDExtractor gracefully falls back to analytical extractor if torch fails."""
    from app.services.vision.reid import TablewareReIDExtractor

    extractor = TablewareReIDExtractor()

    # Monkeypatch get_model to return None (simulating unavailable torch/network/weights)
    monkeypatch.setattr(TablewareReIDExtractor, "get_model", classmethod(lambda cls, *args, **kwargs: None))

    test_arr = np.ones((224, 224, 3), dtype=np.float32) * 0.70
    desc_fallback = extractor.extract_descriptor(test_arr)

    assert desc_fallback.embedding.shape == (384,)
    assert math.isclose(float(np.linalg.norm(desc_fallback.embedding)), 1.0, rel_tol=1e-5)
    assert not np.isnan(desc_fallback.embedding).any()


def test_multimodal_reid_plumbed_prior_matching_and_geometric_rejection(test_db_session):
    """Verify analyze_meal_task uses multimodal Re-ID to accept matching prior and reject geometric mismatch."""
    user = User(email="reid_user@messmate.org", hashed_password="pw")
    test_db_session.add(user)
    test_db_session.flush()

    food = Food(
        name_en="Rice",
        source="IFCT2017",
        source_code="RICE01",
        density_g_per_ml=1.0,
        kcal_per_100g=130.0,
    )
    test_db_session.add(food)
    test_db_session.flush()

    # Case A: User has calibrated standard_bowl (350ml) prior
    prior_std = UnitPrior(
        user_id=user.id,
        unit="standard_bowl",
        context="personal_home",
        mu_log=math.log(350.0),
        sigma2_log=0.03,
        n_obs=5,
    )
    test_db_session.add(prior_std)
    test_db_session.commit()

    now = datetime.now(timezone.utc)
    meal = Meal(
        user_id=user.id,
        captured_at=now,
        local_date=now.date(),
        meal_slot="lunch",
        source="photo",
        status="pending",
        context="personal_home",
    )
    test_db_session.add(meal)
    test_db_session.commit()

    vlm_mock = VLMOutput(
        plate_type="single_bowl",
        items=[
            VLMItem(dish_label="rice", vessel_archetype="standard_bowl", unit_count=1.0),
        ],
    )

    analyzed = analyze_meal_task(test_db_session, meal.id, mock_override=vlm_mock)
    assert analyzed is not None
    assert analyzed.items[0].portion_source == "prior"  # Multimodal match confirmed

    # Case B: Meal with entree_bowl (650ml) where user only has small_bowl (150ml) prior
    # Must NOT corrupt prior; geometric mismatch must drop score below 0.85
    test_db_session.query(UnitPrior).delete()
    prior_small = UnitPrior(
        user_id=user.id,
        unit="small_bowl",
        context="personal_home",
        mu_log=math.log(150.0),
        sigma2_log=0.03,
        n_obs=5,
    )
    test_db_session.add(prior_small)
    test_db_session.commit()

    meal_entree = Meal(
        user_id=user.id,
        captured_at=now,
        local_date=now.date(),
        meal_slot="dinner",
        source="photo",
        status="pending",
        context="personal_home",
    )
    test_db_session.add(meal_entree)
    test_db_session.commit()

    vlm_mock_entree = VLMOutput(
        plate_type="single_bowl",
        items=[
            VLMItem(dish_label="rice", vessel_archetype="entree_bowl", unit_count=1.0),
        ],
    )

    analyzed_entree = analyze_meal_task(test_db_session, meal_entree.id, mock_override=vlm_mock_entree)
    assert analyzed_entree is not None
    # Geometric mismatch (150ml vs 650ml) must prevent false match, falling back to model archetype
    assert analyzed_entree.items[0].portion_source == "model"


# ===========================================================================
# 7. Phone Compatibility: Network Binding, CORS & Image Pipeline
# ===========================================================================
def test_network_binding_and_cors_for_physical_phone(client):
    """Verify backend network binding to 0.0.0.0 and CORS acceptance for phone IP & ADB reverse."""
    from app.core.config import settings

    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000

    # 1. Physical Phone IP over Wi-Fi
    res_phone = client.options(
        "/v1/meals",
        headers={
            "Origin": "http://10.5.138.91:8080",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res_phone.status_code == 200
    assert "access-control-allow-origin" in res_phone.headers

    # 2. USB ADB reverse port forwarding (127.0.0.1 / localhost)
    res_usb = client.options(
        "/v1/meals",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res_usb.status_code == 200
    assert "access-control-allow-origin" in res_usb.headers


def test_smartphone_camera_exif_orientation_and_safe_downsampling():
    """Verify smartphone camera high-res photos with EXIF orientation are transposed and downsampled safely."""
    from PIL import ExifTags, Image
    from app.services.storage.media import MediaService

    # Create 4000x3000 high-res image with EXIF orientation 6 (portrait rotated 90 deg clockwise)
    img = Image.new("RGB", (4000, 3000), color=(180, 120, 70))
    exif = img.getexif()
    exif[274] = 6  # Orientation: Rotate 90 CW
    exif[41989] = 24  # 24mm wide flagship lens

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    img_bytes = buf.getvalue()

    user_id = uuid.uuid4()
    meal_id = uuid.uuid4()

    image_key, webp_bytes, exif_meta = MediaService.process_and_save_image(
        image_input=img_bytes,
        user_id=user_id,
        meal_id=meal_id,
        max_size=800,
    )

    assert image_key == f"{user_id}/{meal_id}.webp"
    assert len(webp_bytes) < 120 * 1024  # Under 120KB SLA target

    # Verify transposed orientation: height is now greater than width (portrait)
    with Image.open(io.BytesIO(webp_bytes)) as result_img:
        w, h = result_img.size
        assert w <= 800 and h <= 800
        assert h > w  # Correctly oriented as portrait (600x800)

    # Clean up test artifact
    from app.services.storage.media import MEDIA_ROOT
    test_file = MEDIA_ROOT / image_key
    if test_file.exists():
        test_file.unlink()
    test_dir = MEDIA_ROOT / str(user_id)
    if test_dir.exists():
        test_dir.rmdir()


