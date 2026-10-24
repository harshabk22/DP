"""Comprehensive test suite for Chunk 2: Nutrition Datasets & 3-Tier Resolution Engine."""

import time
import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models.base import Base
from app.models.food import Food
from app.services.resolution import ResolutionService, ResolvedDish
from scripts.ingest.seed_nutrition_data import CERTIFIED_FOODS, seed_foods


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_seed_nutrition_dataset(db_session):
    """Verifies certified nutrition dataset ingestion.

    Ensures:
    - Exactly 24 real foods ingested from USDA FNDDS, INDB, and IFCT 2017.
    - Zero null or negative calories/macros.
    - Accurate physical bulk densities across food categories.
    - Realistic, normalized 384-D semantic embeddings.
    """
    seeded = seed_foods(db_session)
    assert len(seeded) == 24

    all_foods = list(db_session.scalars(select(Food)).all())
    assert len(all_foods) == 24

    required_fndds = {
        "grilled salmon",
        "grilled chicken breast",
        "cheeseburger",
        "margherita pizza",
        "pasta arrabbiata",
        "caesar salad",
        "oatmeal",
        "latte",
    }
    required_indb = {
        "dal makhani",
        "sambar",
        "rajma",
        "chana masala",
        "chicken tikka",
        "palak paneer",
        "biryani",
        "khichdi",
        "roti/chapati",
        "dosa",
        "idli",
    }
    required_ifct = {
        "brown rice",
        "lentils",
        "spinach",
        "olive oil",
        "ghee",
    }

    names_in_db = {f.name_en for f in all_foods}
    assert required_fndds.issubset(names_in_db), f"Missing FNDDS foods: {required_fndds - names_in_db}"
    assert required_indb.issubset(names_in_db), f"Missing INDB foods: {required_indb - names_in_db}"
    assert required_ifct.issubset(names_in_db), f"Missing IFCT foods: {required_ifct - names_in_db}"

    # Verify nutrient integrity and physical bulk densities
    for food in all_foods:
        assert food.id is not None
        assert food.kcal_per_100g > 0, f"{food.name_en} has non-positive calories"
        assert food.protein_g is not None and food.protein_g >= 0
        assert food.fat_g is not None and food.fat_g >= 0
        assert food.carb_g is not None and food.carb_g >= 0
        assert food.fibre_g is not None and food.fibre_g >= 0
        assert food.density_g_per_ml > 0

        # Verify 384-D normalized vector embedding
        assert food.embedding_json is not None
        assert len(food.embedding_json) == 384
        vec_norm = np.linalg.norm(np.array(food.embedding_json))
        assert abs(vec_norm - 1.0) < 1e-3, f"Embedding for {food.name_en} is not normalized"

        # Check category densities
        name = food.name_en
        density = food.density_g_per_ml
        if name in {"dal makhani", "sambar", "rajma", "chana masala", "latte"}:
            assert 1.00 <= density <= 1.05, f"Density out of bounds for liquid {name}: {density}"
        elif name in {"brown rice", "pasta arrabbiata", "oatmeal", "biryani", "khichdi", "roti/chapati", "dosa", "idli", "lentils", "margherita pizza"}:
            assert 0.80 <= density <= 0.90, f"Density out of bounds for grain {name}: {density}"
        elif name in {"grilled salmon", "grilled chicken breast", "chicken tikka", "palak paneer", "cheeseburger"}:
            assert 1.05 <= density <= 1.15, f"Density out of bounds for dense protein {name}: {density}"
        elif name in {"caesar salad", "spinach"}:
            assert 0.25 <= density <= 0.35, f"Density out of bounds for leafy greens {name}: {density}"
        elif name in {"olive oil", "ghee"}:
            assert abs(density - 0.92) < 1e-3, f"Density out of bounds for fat/oil {name}: {density}"


def test_seed_foods_idempotent(db_session):
    """Verify that seed_foods can be called repeatedly without duplicate error."""
    seed_foods(db_session)
    count_1 = db_session.scalar(select(Food)).id
    seed_foods(db_session)
    all_foods = list(db_session.scalars(select(Food)).all())
    assert len(all_foods) == 24


def test_exact_resolution(db_session):
    """Verifies Tier 1: Normalized exact text & trigram matching (similarity >= 0.90 -> 'exact')."""
    seed_foods(db_session)
    service = ResolutionService(db_session)

    # Exact matches
    exact_targets = ["dal makhani", "sambar", "caesar salad"]
    for q in exact_targets:
        res = service.resolve(q)
        assert res.is_resolved is True
        assert res.resolution_tier == "exact"
        assert res.name_en == q
        assert res.confidence_score >= 0.90
        assert res.kcal_per_100g > 0
        assert res.food_id is not None

    # Case & whitespace normalization
    res_case = service.resolve("   DAL MAKHANI   ")
    assert res_case.name_en == "dal makhani"
    assert res_case.resolution_tier == "exact"
    assert res_case.confidence_score == 1.0

    res_punct = service.resolve("Sambar!")
    assert res_punct.name_en == "sambar"
    assert res_punct.resolution_tier == "exact"


def test_fuzzy_resolution(db_session):
    """Verifies Tier 2: Trigram / token similarity matching (0.70 <= similarity < 0.90 -> 'fuzzy')."""
    seed_foods(db_session)
    service = ResolutionService(db_session)

    typos = [
        ("sambaar", "sambar"),
        ("rajmah", "rajma"),
        ("grild salmon", "grilled salmon"),
    ]

    for typo, expected_target in typos:
        res = service.resolve(typo)
        assert res.is_resolved is True
        assert res.resolution_tier == "fuzzy", f"{typo} was classified as {res.resolution_tier}"
        assert res.name_en == expected_target
        assert 0.70 <= res.confidence_score < 0.90, f"Confidence {res.confidence_score} out of fuzzy bounds"
        assert res.kcal_per_100g > 0
        assert res.food_id is not None


def test_semantic_resolution(db_session):
    """Verifies Tier 3: Semantic vector similarity search (similarity >= 0.60 -> 'fuzzy' or 'decomposed')."""
    seed_foods(db_session)
    service = ResolutionService(db_session)

    # Semantic query "roast salmon fillet" -> matches "grilled salmon"
    res = service.resolve("roast salmon fillet")
    assert res.is_resolved is True
    assert res.name_en == "grilled salmon"
    assert res.resolution_tier in ("fuzzy", "decomposed")
    assert res.confidence_score >= 0.60
    assert res.kcal_per_100g == 206.0
    assert res.density_g_per_ml == 1.08

    # Unresolved fallback
    unresolved_query = "xyzzy_completely_unknown_martian_edible_987"
    res_unres = service.resolve(unresolved_query)
    assert res_unres.is_resolved is False
    assert res_unres.resolution_tier == "unresolved"
    assert res_unres.food_id is None
    assert res_unres.confidence_score < 0.60
    assert res_unres.kcal_per_100g == 0.0


def test_density_mapping(db_session):
    """Verifies physical bulk density taxonomy and volumetric mass calculations.

    Tests that 350ml standard bowl yields physically consistent portion masses:
    Loose salads (0.25 - 0.35 g/ml) < Grains/Rice (0.80 - 0.90 g/ml) < Dal/Liquids (1.00 - 1.05 g/ml).
    """
    seed_foods(db_session)
    service = ResolutionService(db_session)

    salad = service.resolve("caesar salad")
    dal = service.resolve("dal makhani")
    rice = service.resolve("brown rice")

    assert 0.25 <= salad.density_g_per_ml <= 0.35
    assert 1.00 <= dal.density_g_per_ml <= 1.05
    assert 0.80 <= rice.density_g_per_ml <= 0.90

    # Volumetric mass test for 350ml standard bowl
    bowl_volume_ml = 350.0
    mass_salad = bowl_volume_ml * salad.density_g_per_ml
    mass_rice = bowl_volume_ml * rice.density_g_per_ml
    mass_dal = bowl_volume_ml * dal.density_g_per_ml

    # Physical reality invariants
    assert mass_salad < mass_rice < mass_dal
    assert 85.0 <= mass_salad <= 125.0, f"Salad mass unexpected: {mass_salad}"
    assert 280.0 <= mass_rice <= 315.0, f"Rice mass unexpected: {mass_rice}"
    assert 350.0 <= mass_dal <= 370.0, f"Dal mass unexpected: {mass_dal}"

    # Density ratio: Dal is ~3x denser than loose leafy salad
    density_ratio = dal.density_g_per_ml / salad.density_g_per_ml
    assert density_ratio >= 3.0


def test_resolution_latency(db_session):
    """Verifies 100 consecutive lookups complete in < 15ms average per lookup."""
    seed_foods(db_session)
    service = ResolutionService(db_session)

    test_queries = [
        "dal makhani",
        "sambar",
        "caesar salad",
        "sambaar",
        "rajmah",
        "grild salmon",
        "roast salmon fillet",
        "tandoori chicken pieces",
        "steamed brown rice",
        "unknown exotic dish 404",
    ] * 10  # 100 queries

    assert len(test_queries) == 100

    start_time = time.perf_counter()
    results = [service.resolve(q) for q in test_queries]
    total_elapsed = time.perf_counter() - start_time

    avg_ms = (total_elapsed / len(test_queries)) * 1000.0

    print(f"\n100 lookups completed in {total_elapsed * 1000.0:.2f} ms ({avg_ms:.3f} ms/lookup)")
    assert len(results) == 100
    assert avg_ms < 15.0, f"Resolution latency exceeded 15ms limit: {avg_ms:.3f} ms"
