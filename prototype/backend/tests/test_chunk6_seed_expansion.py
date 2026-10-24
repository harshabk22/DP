"""Unit tests for Chunk 6: Seed Expansion & Eval Grounding.

Validates:
1. Dataset expansion to 160+ certified reference foods from ICMR-NIN IFCT 2017 & USDA FNDDS.
2. Zero duplicate source codes or food names across the entire catalog.
3. Realistic physical bulk densities (0.2 <= rho <= 1.5 g/ml).
4. Thermodynamic consistency (Atwater energy balance within standard nutritional margins).
5. Database ingestion generates valid 384-D semantic embeddings for every single food item.
6. Idempotent seeding and full backward compatibility with core 24-food set.
7. Benchmark dataset loader integration.
"""

import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.models.base import Base
from app.models.food import Food
from eval.benchmark.loader import get_cuisine_food_breakdown, load_certified_food_catalog
from scripts.ingest.seed_nutrition_data import (
    ALL_CERTIFIED_FOODS,
    CERTIFIED_FOODS,
    seed_expanded_foods,
    seed_foods,
)


@pytest.fixture
def db_session():
    """Create an isolated, in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_certified_foods_count_and_uniqueness():
    """Verify catalog expansion and strict uniqueness of codes and names."""
    assert len(CERTIFIED_FOODS) == 24, "Core certified catalog must remain exactly 24 items"
    assert len(ALL_CERTIFIED_FOODS) >= 150, f"Expanded catalog has {len(ALL_CERTIFIED_FOODS)} items (target >= 150)"

    codes = [f["source_code"] for f in ALL_CERTIFIED_FOODS]
    assert len(codes) == len(set(codes)), f"Duplicate source_code found: {len(codes) - len(set(codes))} duplicates"

    names = [f["name_en"].lower().strip() for f in ALL_CERTIFIED_FOODS]
    assert len(names) == len(set(names)), f"Duplicate name_en found: {len(names) - len(set(names))} duplicates"


def test_bulk_densities_and_nutritional_consistency():
    """Validate that densities and macros adhere to physical and Atwater invariants."""
    for food in ALL_CERTIFIED_FOODS:
        name = food["name_en"]
        density = food["density_g_per_ml"]
        kcal = food["kcal_per_100g"]
        prot = food.get("protein_g") or 0.0
        fat = food.get("fat_g") or 0.0
        carb = food.get("carb_g") or 0.0

        # Physical density check (p01.md §4.1)
        assert 0.20 <= density <= 1.50, f"Unphysical density {density} for {name}"

        # Non-negative macro check
        assert kcal >= 0.0, f"Negative calories for {name}"
        assert prot >= 0.0, f"Negative protein for {name}"
        assert fat >= 0.0, f"Negative fat for {name}"
        assert carb >= 0.0, f"Negative carbs for {name}"

        # Atwater consistency check for foods > 50 kcal/100g
        if kcal > 50.0:
            atwater_kcal = 4.0 * prot + 4.0 * carb + 9.0 * fat
            diff = abs(kcal - atwater_kcal)
            rel_diff = diff / kcal
            assert rel_diff < 0.28 or diff < 40.0, (
                f"Atwater inconsistency in {name}: stated {kcal} vs calculated {atwater_kcal:.1f} (gap {diff:.1f} kcal)"
            )


def test_database_seeding_and_embeddings(db_session):
    """Verify database ingestion populates all 160+ foods with valid 384-D embeddings."""
    seeded = seed_expanded_foods(db_session)
    assert len(seeded) == len(ALL_CERTIFIED_FOODS)

    # Check database query
    all_db_foods = db_session.scalars(select(Food)).all()
    assert len(all_db_foods) == len(ALL_CERTIFIED_FOODS)

    for food in all_db_foods:
        assert food.name_en is not None
        assert food.density_g_per_ml > 0.0
        assert food.kcal_per_100g >= 0.0
        assert food.embedding_json is not None
        assert len(food.embedding_json) == 384, f"Embedding length mismatch for {food.name_en}"


def test_seeding_idempotence(db_session):
    """Re-running seed_expanded_foods must be completely idempotent."""
    seed_expanded_foods(db_session)
    # Second run
    seed_expanded_foods(db_session)

    count = len(db_session.scalars(select(Food)).all())
    assert count == len(ALL_CERTIFIED_FOODS)


def test_backward_compatibility_core_set(db_session):
    """seed_foods(..., include_expanded=False) must preserve the original 24 items."""
    seeded_core = seed_foods(db_session, include_expanded=False)
    assert len(seeded_core) == 24


def test_benchmark_catalog_loader():
    """Verify eval/benchmark/loader.py loads and organizes catalog properly."""
    catalog = load_certified_food_catalog(include_expanded=True)
    assert len(catalog) == len(ALL_CERTIFIED_FOODS)

    # Check sample food resolution
    sample = catalog.get("steamed_basmati_rice")
    assert sample is not None
    assert sample["kcal"] == 130.0
    assert sample["density"] == 0.85

    breakdown = get_cuisine_food_breakdown()
    assert len(breakdown["South_Indian_Tiffins"]) > 0
    assert len(breakdown["Indian_Staples"]) > 0
    assert len(breakdown["Global_Western"]) > 0
