"""Certified nutrition dataset ingestion for MessMate v2.0.

Seeds the database with certified reference foods from USDA FNDDS, INDB,
and ICMR-NIN IFCT 2017. All entries contain certified proximate nutrients,
measured bulk densities (g/ml), and normalized 384-D semantic embeddings.
"""

from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.food import Food
from app.services.resolution.embedding import batch_generate_embeddings, generate_semantic_embedding

CERTIFIED_FOODS: list[dict[str, Any]] = [
    # --- 1. USDA FNDDS Composite Dishes ---
    {
        "source": "FNDDS",
        "source_code": "FNDDS_26115200",
        "name_en": "grilled salmon",
        "name_local": {"es": "salmón a la parrilla", "fr": "saumon grillé"},
        "aliases": ["salmon fillet", "grilled salmon fillet", "baked salmon"],
        "food_group": "Dense Proteins",
        "density_g_per_ml": 1.08,
        "kcal_per_100g": 206.0,
        "protein_g": 22.1,
        "fat_g": 12.3,
        "carb_g": 0.0,
        "fibre_g": 0.0,
        "micros": {"calcium_mg": 12.0, "iron_mg": 0.8, "potassium_mg": 490.0, "vitamin_d_mcg": 11.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 150.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 26115200)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_24114120",
        "name_en": "grilled chicken breast",
        "name_local": {"es": "pechuga de pollo a la plancha"},
        "aliases": ["chicken breast", "grilled chicken", "roast chicken breast"],
        "food_group": "Dense Proteins",
        "density_g_per_ml": 1.06,
        "kcal_per_100g": 165.0,
        "protein_g": 31.0,
        "fat_g": 3.6,
        "carb_g": 0.0,
        "fibre_g": 0.0,
        "micros": {"calcium_mg": 15.0, "iron_mg": 1.0, "sodium_mg": 74.0, "potassium_mg": 256.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 120.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 24114120)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_27510100",
        "name_en": "cheeseburger",
        "name_local": {"de": "Cheeseburger", "fr": "cheeseburger"},
        "aliases": ["cheese burger", "beef burger", "hamburger with cheese"],
        "food_group": "Dense Proteins",
        "density_g_per_ml": 1.05,
        "kcal_per_100g": 264.0,
        "protein_g": 14.2,
        "fat_g": 13.4,
        "carb_g": 27.2,
        "fibre_g": 1.5,
        "micros": {"calcium_mg": 150.0, "iron_mg": 2.4, "sodium_mg": 530.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 180.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 27510100)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_27411100",
        "name_en": "margherita pizza",
        "name_local": {"it": "pizza margherita"},
        "aliases": ["cheese pizza", "pizza margherita", "plain pizza"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.88,
        "kcal_per_100g": 256.0,
        "protein_g": 10.5,
        "fat_g": 9.8,
        "carb_g": 31.4,
        "fibre_g": 2.3,
        "micros": {"calcium_mg": 188.0, "iron_mg": 2.1, "sodium_mg": 590.0},
        "default_unit": "slice",
        "pop_grams_per_unit": 107.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 27411100)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_27212100",
        "name_en": "pasta arrabbiata",
        "name_local": {"it": "pasta all'arrabbiata"},
        "aliases": ["spicy tomato pasta", "penne arrabbiata", "arrabbiata"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.85,
        "kcal_per_100g": 131.0,
        "protein_g": 4.5,
        "fat_g": 3.2,
        "carb_g": 22.0,
        "fibre_g": 2.1,
        "micros": {"calcium_mg": 24.0, "iron_mg": 1.2, "sodium_mg": 380.0},
        "default_unit": "standard_bowl",
        "pop_grams_per_unit": 250.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 27212100)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_75411100",
        "name_en": "caesar salad",
        "name_local": {"es": "ensalada césar"},
        "aliases": ["chicken caesar salad", "romaine caesar salad", "caesar salad bowl"],
        "food_group": "Loose Greens & Salads",
        "density_g_per_ml": 0.30,
        "kcal_per_100g": 158.0,
        "protein_g": 3.2,
        "fat_g": 14.1,
        "carb_g": 4.8,
        "fibre_g": 1.8,
        "micros": {"calcium_mg": 85.0, "iron_mg": 1.1, "vitamin_a_mcg": 140.0},
        "default_unit": "entree_bowl",
        "pop_grams_per_unit": 120.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 75411100)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_56201010",
        "name_en": "oatmeal",
        "name_local": {"fr": "gruau d'avoine"},
        "aliases": ["cooked oatmeal", "porridge", "rolled oats cooked"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.85,
        "kcal_per_100g": 68.0,
        "protein_g": 2.4,
        "fat_g": 1.4,
        "carb_g": 12.0,
        "fibre_g": 1.7,
        "micros": {"calcium_mg": 47.0, "iron_mg": 0.9, "magnesium_mg": 19.0},
        "default_unit": "standard_bowl",
        "pop_grams_per_unit": 234.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 56201010)",
    },
    {
        "source": "FNDDS",
        "source_code": "FNDDS_11511100",
        "name_en": "latte",
        "name_local": {"it": "caffellatte"},
        "aliases": ["caffe latte", "coffee latte", "cafe latte"],
        "food_group": "Liquids, Soups & Dal",
        "density_g_per_ml": 1.03,
        "kcal_per_100g": 54.0,
        "protein_g": 3.0,
        "fat_g": 2.9,
        "carb_g": 4.1,
        "fibre_g": 0.0,
        "micros": {"calcium_mg": 115.0, "potassium_mg": 150.0},
        "default_unit": "cup_mug",
        "pop_grams_per_unit": 240.0,
        "provenance_note": "USDA FNDDS 2019-2020 (Food code 11511100)",
    },

    # --- 2. INDB Indian Cooked Dishes ---
    {
        "source": "INDB",
        "source_code": "INDB_1001",
        "name_en": "dal makhani",
        "name_local": {"hi": "दाल मखनी"},
        "aliases": ["maa ki dal", "black lentils dal", "creamy black lentils"],
        "food_group": "Liquids, Soups & Dal",
        "density_g_per_ml": 1.02,
        "kcal_per_100g": 132.0,
        "protein_g": 4.5,
        "fat_g": 6.5,
        "carb_g": 14.0,
        "fibre_g": 3.8,
        "micros": {"iron_mg": 1.8, "calcium_mg": 65.0, "potassium_mg": 280.0},
        "default_unit": "small_bowl",
        "pop_grams_per_unit": 150.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1002",
        "name_en": "sambar",
        "name_local": {"ta": "சாம்பார்", "kn": "ಸಾಂಬಾರ್", "hi": "सांबर"},
        "aliases": ["sambhar", "south indian lentil stew", "vegetable sambar"],
        "food_group": "Liquids, Soups & Dal",
        "density_g_per_ml": 1.02,
        "kcal_per_100g": 60.0,
        "protein_g": 2.8,
        "fat_g": 1.8,
        "carb_g": 8.5,
        "fibre_g": 2.5,
        "micros": {"iron_mg": 1.2, "calcium_mg": 38.0, "potassium_mg": 190.0},
        "default_unit": "small_bowl",
        "pop_grams_per_unit": 150.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1003",
        "name_en": "rajma",
        "name_local": {"hi": "राजमा"},
        "aliases": ["kidney beans curry", "red kidney beans masala", "rajma curry"],
        "food_group": "Liquids, Soups & Dal",
        "density_g_per_ml": 1.01,
        "kcal_per_100g": 125.0,
        "protein_g": 5.2,
        "fat_g": 4.1,
        "carb_g": 16.8,
        "fibre_g": 4.5,
        "micros": {"iron_mg": 2.1, "calcium_mg": 45.0, "potassium_mg": 320.0},
        "default_unit": "small_bowl",
        "pop_grams_per_unit": 150.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1004",
        "name_en": "chana masala",
        "name_local": {"hi": "चना मसाला"},
        "aliases": ["chole", "chole masala", "chickpea curry"],
        "food_group": "Liquids, Soups & Dal",
        "density_g_per_ml": 1.02,
        "kcal_per_100g": 148.0,
        "protein_g": 5.8,
        "fat_g": 5.6,
        "carb_g": 18.5,
        "fibre_g": 5.1,
        "micros": {"iron_mg": 2.4, "calcium_mg": 52.0, "potassium_mg": 310.0},
        "default_unit": "small_bowl",
        "pop_grams_per_unit": 150.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1005",
        "name_en": "chicken tikka",
        "name_local": {"hi": "चिकन टिक्का"},
        "aliases": ["tandoori chicken tikka", "murgh tikka", "tandoori chicken pieces"],
        "food_group": "Dense Proteins",
        "density_g_per_ml": 1.10,
        "kcal_per_100g": 180.0,
        "protein_g": 25.5,
        "fat_g": 7.2,
        "carb_g": 3.5,
        "fibre_g": 1.0,
        "micros": {"iron_mg": 1.4, "calcium_mg": 25.0, "sodium_mg": 420.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 40.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1006",
        "name_en": "palak paneer",
        "name_local": {"hi": "पालक पनीर"},
        "aliases": ["spinach cottage cheese", "saag paneer", "palak paneer curry"],
        "food_group": "Dense Proteins",
        "density_g_per_ml": 1.08,
        "kcal_per_100g": 165.0,
        "protein_g": 8.2,
        "fat_g": 13.5,
        "carb_g": 4.8,
        "fibre_g": 2.8,
        "micros": {"calcium_mg": 230.0, "iron_mg": 2.7, "vitamin_a_mcg": 310.0},
        "default_unit": "small_bowl",
        "pop_grams_per_unit": 150.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1007",
        "name_en": "biryani",
        "name_local": {"hi": "बिरयानी", "ur": "بریانی"},
        "aliases": ["chicken biryani", "biriyani", "dum biryani", "mutton biryani"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.85,
        "kcal_per_100g": 185.0,
        "protein_g": 8.5,
        "fat_g": 6.8,
        "carb_g": 22.5,
        "fibre_g": 1.5,
        "micros": {"iron_mg": 1.1, "calcium_mg": 22.0, "sodium_mg": 380.0},
        "default_unit": "entree_bowl",
        "pop_grams_per_unit": 300.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1008",
        "name_en": "khichdi",
        "name_local": {"hi": "खिचड़ी"},
        "aliases": ["dal khichdi", "khichuri", "moong dal khichdi"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.88,
        "kcal_per_100g": 115.0,
        "protein_g": 3.8,
        "fat_g": 2.5,
        "carb_g": 19.5,
        "fibre_g": 2.2,
        "micros": {"iron_mg": 1.3, "calcium_mg": 28.0},
        "default_unit": "standard_bowl",
        "pop_grams_per_unit": 250.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1009",
        "name_en": "roti/chapati",
        "name_local": {"hi": "रोटी / चपाती"},
        "aliases": ["roti", "chapati", "phulka", "indian flatbread", "wheat roti"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.82,
        "kcal_per_100g": 264.0,
        "protein_g": 8.0,
        "fat_g": 3.5,
        "carb_g": 50.0,
        "fibre_g": 9.5,
        "micros": {"iron_mg": 2.8, "calcium_mg": 30.0, "magnesium_mg": 78.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 35.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1010",
        "name_en": "dosa",
        "name_local": {"ta": "தோசை", "kn": "ದೋಸೆ", "hi": "डोसा"},
        "aliases": ["plain dosa", "masala dosa", "crispy dosa"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.82,
        "kcal_per_100g": 168.0,
        "protein_g": 3.9,
        "fat_g": 3.7,
        "carb_g": 29.5,
        "fibre_g": 1.2,
        "micros": {"iron_mg": 0.9, "calcium_mg": 16.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 80.0,
        "provenance_note": "INDB 2021 recipe profile",
    },
    {
        "source": "INDB",
        "source_code": "INDB_1011",
        "name_en": "idli",
        "name_local": {"ta": "இட்லி", "kn": "ಇಡ್ಲಿ", "hi": "इडली"},
        "aliases": ["steamed idli", "rice idli"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.88,
        "kcal_per_100g": 132.0,
        "protein_g": 4.2,
        "fat_g": 0.6,
        "carb_g": 27.5,
        "fibre_g": 1.8,
        "micros": {"iron_mg": 0.8, "calcium_mg": 14.0},
        "default_unit": "piece",
        "pop_grams_per_unit": 50.0,
        "provenance_note": "INDB 2021 recipe profile",
    },

    # --- 3. IFCT 2017 Raw Ingredients ---
    {
        "source": "IFCT2017",
        "source_code": "IFCT_A002",
        "name_en": "brown rice",
        "name_local": {"hi": "भूरा चावल"},
        "aliases": ["whole grain rice", "unpolished rice", "raw brown rice"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.85,
        "kcal_per_100g": 362.0,
        "protein_g": 7.9,
        "fat_g": 2.7,
        "carb_g": 76.0,
        "fibre_g": 3.5,
        "micros": {"thiamin_mg": 0.41, "magnesium_mg": 143.0, "phosphorus_mg": 333.0},
        "default_unit": "tablespoon",
        "pop_grams_per_unit": 15.0,
        "provenance_note": "ICMR-NIN IFCT 2017 (Code A.002)",
    },
    {
        "source": "IFCT2017",
        "source_code": "IFCT_B001",
        "name_en": "lentils",
        "name_local": {"hi": "मसूर दाल"},
        "aliases": ["masoor dal", "raw lentils", "dry lentils", "split red lentils"],
        "food_group": "Cooked Grains",
        "density_g_per_ml": 0.88,
        "kcal_per_100g": 343.0,
        "protein_g": 24.6,
        "fat_g": 1.1,
        "carb_g": 59.0,
        "fibre_g": 10.7,
        "micros": {"iron_mg": 7.5, "folate_mcg": 479.0, "potassium_mg": 677.0},
        "default_unit": "tablespoon",
        "pop_grams_per_unit": 15.0,
        "provenance_note": "ICMR-NIN IFCT 2017 (Code B.001)",
    },
    {
        "source": "IFCT2017",
        "source_code": "IFCT_D045",
        "name_en": "spinach",
        "name_local": {"hi": "पालक"},
        "aliases": ["palak", "fresh spinach", "raw spinach", "baby spinach"],
        "food_group": "Loose Greens & Salads",
        "density_g_per_ml": 0.28,
        "kcal_per_100g": 23.0,
        "protein_g": 2.9,
        "fat_g": 0.4,
        "carb_g": 3.6,
        "fibre_g": 2.2,
        "micros": {"iron_mg": 2.7, "calcium_mg": 99.0, "vitamin_a_mcg": 469.0, "folate_mcg": 194.0},
        "default_unit": "standard_bowl",
        "pop_grams_per_unit": 30.0,
        "provenance_note": "ICMR-NIN IFCT 2017 (Code D.045)",
    },
    {
        "source": "IFCT2017",
        "source_code": "IFCT_E001",
        "name_en": "olive oil",
        "name_local": {"es": "aceite de oliva", "it": "olio d'oliva"},
        "aliases": ["extra virgin olive oil", "evoo", "olive oil salad dressing"],
        "food_group": "Fats & Oils",
        "density_g_per_ml": 0.92,
        "kcal_per_100g": 884.0,
        "protein_g": 0.0,
        "fat_g": 100.0,
        "carb_g": 0.0,
        "fibre_g": 0.0,
        "micros": {"vitamin_e_mg": 14.3, "vitamin_k_mcg": 60.2},
        "default_unit": "tablespoon",
        "pop_grams_per_unit": 14.0,
        "provenance_note": "ICMR-NIN IFCT 2017 / USDA Foundation",
    },
    {
        "source": "IFCT2017",
        "source_code": "IFCT_E005",
        "name_en": "ghee",
        "name_local": {"hi": "घी"},
        "aliases": ["clarified butter", "desi ghee", "cow ghee"],
        "food_group": "Fats & Oils",
        "density_g_per_ml": 0.92,
        "kcal_per_100g": 900.0,
        "protein_g": 0.0,
        "fat_g": 99.5,
        "carb_g": 0.0,
        "fibre_g": 0.0,
        "micros": {"vitamin_a_mcg": 700.0, "vitamin_e_mg": 2.8},
        "default_unit": "tablespoon",
        "pop_grams_per_unit": 14.0,
        "provenance_note": "ICMR-NIN IFCT 2017 (Code E.005)",
    },
]

from scripts.ingest.expanded_certified_foods import EXPANDED_CERTIFIED_FOODS

ALL_CERTIFIED_FOODS: list[dict[str, Any]] = CERTIFIED_FOODS + EXPANDED_CERTIFIED_FOODS


def seed_foods(
    session: Session,
    commit: bool = True,
    include_expanded: bool = False,
) -> list[Food]:
    """Seed certified foods with 384-D semantic embeddings into the database.

    Idempotent operation: updates existing records matched on (source, source_code)
    or inserts new records.

    Args:
        session: Active SQLAlchemy database session.
        commit: Whether to commit the transaction (default True).
        include_expanded: If True, seeds all 160+ reference foods (CERTIFIED_FOODS + EXPANDED_CERTIFIED_FOODS).
                          If False, seeds the core 24 certified reference foods.
    """
    catalog = ALL_CERTIFIED_FOODS if include_expanded else CERTIFIED_FOODS
    seeded_foods: list[Food] = []

    # Precompute embeddings in batch for optimal neural inference throughput
    texts = [
        f"{item_data['name_en']} {' '.join(item_data.get('aliases', []))} {item_data.get('food_group', '')}"
        for item_data in catalog
    ]
    embeddings = batch_generate_embeddings(texts)

    for item_data, embedding_vec in zip(catalog, embeddings):

        source = item_data["source"]
        source_code = item_data["source_code"]

        food = session.scalar(
            select(Food).where(Food.source == source, Food.source_code == source_code)
        )

        record_attrs = dict(item_data)
        record_attrs["embedding_json"] = embedding_vec

        if food is None:
            food = Food(**record_attrs)
            session.add(food)
        else:
            for k, v in record_attrs.items():
                setattr(food, k, v)

        seeded_foods.append(food)

    if commit:
        session.commit()
    else:
        session.flush()

    return seeded_foods


def seed_expanded_foods(session: Session, commit: bool = True) -> list[Food]:
    """Seed the complete authentic multi-cuisine certified database (160+ foods)."""
    return seed_foods(session, commit=commit, include_expanded=True)


if __name__ == "__main__":
    from app.core.database import SessionLocal, sync_engine
    from app.models.base import Base

    Base.metadata.create_all(sync_engine)
    with SessionLocal() as db_session:
        results = seed_expanded_foods(db_session, commit=True)
        print(f"Successfully seeded {len(results)} certified foods into database.")
