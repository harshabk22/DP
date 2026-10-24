"""Dataset loader connecting Tier 1 benchmark to verified food composition catalog.

Adheres to MESSMATE_SPEC.md §13.1 & Invariant I1:
- Zero generative LLMs in nutrient computation.
- Connects benchmark runner to ICMR-NIN IFCT 2017 & USDA FNDDS certified reference items (166 foods).
"""

from typing import Any, Dict, List
import sys
from pathlib import Path

# Ensure backend directory is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from scripts.ingest.seed_nutrition_data import ALL_CERTIFIED_FOODS, CERTIFIED_FOODS


def load_certified_food_catalog(include_expanded: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load certified foods mapped by normalized lookup slug.

    Args:
        include_expanded: If True, returns all 166 foods; if False, returns the core 24 foods.

    Returns:
        Dict mapping food slug -> nutrient density dictionary:
            {"name_en": str, "kcal": float, "protein": float, "fat": float, "carb": float,
             "density": float, "fibre": float, "source": str, "food_group": str}
    """
    catalog_source = ALL_CERTIFIED_FOODS if include_expanded else CERTIFIED_FOODS
    catalog: Dict[str, Dict[str, Any]] = {}

    for item in catalog_source:
        slug = item["name_en"].lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        catalog[slug] = {
            "name_en": item["name_en"],
            "kcal": float(item["kcal_per_100g"]),
            "protein": float(item.get("protein_g") or 0.0),
            "fat": float(item.get("fat_g") or 0.0),
            "carb": float(item.get("carb_g") or 0.0),
            "density": float(item["density_g_per_ml"]),
            "fibre": float(item.get("fibre_g") or 0.0),
            "source": item.get("source", "IFCT2017"),
            "food_group": item.get("food_group", ""),
        }

    return catalog


def load_benchmark_catalog(include_expanded: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load authentic certified benchmark catalog (166 foods) for Tier 1 evaluation."""
    return load_certified_food_catalog(include_expanded=include_expanded)


def get_cuisine_food_breakdown() -> Dict[str, list[str]]:
    """Group certified catalog items by dietary / cuisine family."""
    catalog = load_certified_food_catalog(include_expanded=True)
    breakdown: Dict[str, list[str]] = {
        "Indian_Staples": [],
        "Global_Western": [],
        "Dairy_and_Desserts": [],
        "South_Indian_Tiffins": [],
    }

    for slug in catalog:
        if any(w in slug for w in ["dosa", "idli", "upma", "pongal", "uttapam", "appam"]):
            breakdown["South_Indian_Tiffins"].append(slug)
        elif any(w in slug for w in ["curd", "lassi", "kheer", "halwa", "jamun", "rasgulla", "shrikhand"]):
            breakdown["Dairy_and_Desserts"].append(slug)
        elif any(w in slug for w in ["burger", "pizza", "pasta", "steak", "caesar", "salmon"]):
            breakdown["Global_Western"].append(slug)
        else:
            breakdown["Indian_Staples"].append(slug)

    return breakdown


def get_tier1_cuisines() -> Dict[str, list[str]]:
    """Partition the 166 certified reference foods into the 4 Tier 1 cuisines (§13.1).

    Returns:
        Dict mapping cuisine name -> list of food slugs:
            {"Western": [...], "Asian": [...], "Indian": [...], "Mediterranean": [...]}
    """
    asian_keywords = [
        "pad thai", "ramen", "dim sum", "pho", "bibimbap", "kung pao",
        "jasmine rice", "fried rice", "miso soup",
    ]
    med_keywords = [
        "greek salad", "hummus", "pita", "souvlaki", "falafel", "tabbouleh",
        "shakshuka", "sea bass", "couscous", "tzatziki",
    ]
    western_keywords = [
        "burger", "pizza", "pasta", "steak", "caesar", "salmon", "broccoli",
        "potatoes", "oatmeal", "macaroni", "french fries", "sandwich",
        "grilled chicken", "quinoa", "scrambled egg", "toast", "latte",
    ]

    cuisines: Dict[str, list[str]] = {
        "Western": [],
        "Asian": [],
        "Indian": [],
        "Mediterranean": [],
    }

    for item in ALL_CERTIFIED_FOODS:
        name = item["name_en"].lower()
        slug = name.replace(" ", "_").replace("/", "_").replace("-", "_")
        fg = item.get("food_group", "")

        if fg == "Asian Dishes" or any(k in name for k in asian_keywords):
            cuisines["Asian"].append(slug)
        elif fg == "Mediterranean" or any(k in name for k in med_keywords):
            cuisines["Mediterranean"].append(slug)
        elif any(k in name for k in western_keywords) or (
            item.get("source") == "FNDDS"
            and not any(k in name for k in ["curry", "tikka", "biryani", "dal", "paneer"])
        ):
            cuisines["Western"].append(slug)
        else:
            cuisines["Indian"].append(slug)

    return cuisines
