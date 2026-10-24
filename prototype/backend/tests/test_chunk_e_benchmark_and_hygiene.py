"""Phase 3 Chunk E Tests: Scientific Benchmark Catalog Grounding & Codebase Hygiene.

Validates:
1. Grounding of eval/benchmark/tier1_runner in ALL_CERTIFIED_FOODS (>= 150 items, 166 total).
2. Dynamic synthesis of 200 weighed multi-ingredient dishes across 4 cuisines using certified data.
3. Scientific benchmark thresholds: Calorie MAE reduction >= 40% (p < 0.0001) and macro bounds.
4. Health endpoint cleanliness and prefix parity across /health, /v1/health, /api/v1/health, /api/health with status "ok" and DB connectivity.
"""

from pathlib import Path
import sys
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

# Ensure project root and backend are on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TEST_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.api.deps import get_db
from app.core.database import SessionLocal, sync_engine
from app.main import app
from app.models.base import Base
from app.models.triggers import register_i5_triggers
from eval.benchmark.loader import (
    ALL_CERTIFIED_FOODS,
    get_tier1_cuisines,
    load_benchmark_catalog,
)
from eval.benchmark.tier1_runner import (
    CUISINE_KEYS,
    FOOD_CATALOG,
    generate_tier1_dataset,
    run_tier1_benchmark,
)


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    register_i5_triggers(engine)

    SessionLocal.configure(bind=engine)

    with Session(engine) as session:
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


def test_certified_catalog_grounding():
    """Verify that tier1_runner and benchmark loader operate over ALL_CERTIFIED_FOODS (>= 150 items)."""
    assert len(ALL_CERTIFIED_FOODS) >= 150, f"Expected >= 150 certified foods, got {len(ALL_CERTIFIED_FOODS)}"
    assert len(FOOD_CATALOG) >= 150, f"Expected >= 150 catalog foods, got {len(FOOD_CATALOG)}"
    assert len(FOOD_CATALOG) == len(ALL_CERTIFIED_FOODS)

    # Check key physical properties on certified catalog items
    for slug, item in FOOD_CATALOG.items():
        assert item["kcal"] > 0, f"Food {slug} has non-positive calories"
        assert item["density"] > 0, f"Food {slug} has non-positive density"
        assert "protein" in item
        assert "fat" in item
        assert "carb" in item
        assert "fibre" in item


def test_cuisine_partition_coverage():
    """Verify all 4 Tier 1 cuisines (Western, Asian, Indian, Mediterranean) have certified items."""
    cuisines = get_tier1_cuisines()
    assert set(cuisines.keys()) == {"Western", "Asian", "Indian", "Mediterranean"}

    total_partitioned = 0
    for cuisine_name, items in cuisines.items():
        assert len(items) >= 8, f"Cuisine {cuisine_name} has insufficient items ({len(items)})"
        total_partitioned += len(items)
        # Verify all items in cuisine exist in FOOD_CATALOG
        for item_key in items:
            assert item_key in FOOD_CATALOG, f"Missing {item_key} in FOOD_CATALOG"

    assert total_partitioned == len(ALL_CERTIFIED_FOODS)


def test_tier1_dataset_generation_properties():
    """Verify synthesized 200 weighed dishes draw from authentic certified items with realistic portions."""
    dishes = generate_tier1_dataset(seed=42)
    assert len(dishes) == 200, f"Expected 200 dishes, got {len(dishes)}"

    cuisine_counts = {}
    for dish in dishes:
        c = dish["cuisine"]
        cuisine_counts[c] = cuisine_counts.get(c, 0) + 1
        assert 2 <= len(dish["components"]) <= 4, "Each dish must have 2 to 4 components"

        # Check scale precision and component nutritional fidelity
        gt = dish["ground_truth"]
        calc_kcal = 0.0
        calc_mass = 0.0
        for comp in dish["components"]:
            assert comp["food_key"] in FOOD_CATALOG
            assert 60.0 <= comp["mass_g"] <= 260.0
            assert comp["density"] > 0.0
            calc_kcal += comp["kcal"]
            calc_mass += comp["mass_g"]

        assert abs(gt["mass_g"] - calc_mass) < 0.5
        assert abs(gt["kcal"] - calc_kcal) < 0.5

    for c in ["Western", "Asian", "Indian", "Mediterranean"]:
        assert cuisine_counts[c] == 50, f"Expected 50 dishes for {c}, got {cuisine_counts.get(c)}"


def test_tier1_benchmark_scientific_thresholds():
    """Verify that Tier 1 benchmark achieves >= 40% reduction in calorie MAE (p < 0.0001) and passes macro bounds."""
    results = run_tier1_benchmark(seed=42)

    overall = results["overall"]
    assert "kcal" in overall
    assert "protein_g" in overall
    assert "fat_g" in overall
    assert "carb_g" in overall

    # Calorie MAE reduction >= 40% and p < 0.0001
    cal_red = overall["kcal"]["mae_reduction_pct"]
    assert cal_red >= 40.0, f"Expected Calorie MAE reduction >= 40%, got {cal_red}%"
    assert overall["kcal"]["p_value"] == "< 0.0001", f"Expected p < 0.0001, got {overall['kcal']['p_value']}"
    assert overall["kcal"]["paired_t_statistic"] > 3.29

    # Macronutrient bounds
    assert overall["protein_g"]["mae_reduction_pct"] >= 40.0
    assert overall["fat_g"]["mae_reduction_pct"] >= 40.0
    assert overall["carb_g"]["mae_reduction_pct"] >= 40.0

    # Verify results export
    results_json = _PROJECT_ROOT / "eval" / "results" / "tier1_weighed_benchmark_results.json"
    assert results_json.exists(), "Results JSON was not generated"


def test_health_endpoints_status_ok_across_all_prefixes(client):
    """Verify health endpoints respond with status: ok across /health, /v1/health, /api/v1/health, /api/health."""
    endpoints = ["/health", "/v1/health", "/api/v1/health", "/api/health"]
    for ep in endpoints:
        res = client.get(ep)
        assert res.status_code == 200, f"Expected 200 from {ep}, got {res.status_code}"
        data = res.json()
        assert data.get("status") == "ok", f"Expected status 'ok' from {ep}, got {data}"
        assert data.get("version") == "2.0.0"

    # Deep health probe at /v1/health must include DB connectivity status
    v1_res = client.get("/v1/health")
    assert v1_res.json().get("database") == "connected"
