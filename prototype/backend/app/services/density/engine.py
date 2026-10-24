"""Dedicated Culinary Bulk Density Reference Service.

Adheres to MESSMATE_SPEC.md §6.2, §13.3 and Master Blueprint §2:
Provides certified physical bulk densities (rho in g/ml) for volume-to-mass
conversion across the South Asian and international culinary taxonomy.
Decouples latent tableware volume (ml) from food mass (grams):
    m_i = V_i * rho_d
"""

from enum import Enum
import re
from typing import NamedTuple, Optional
from pydantic import BaseModel, ConfigDict, Field


class DensityCategory(str, Enum):
    GRAINS = "grains"                # Cooked rice, biryani, khichdi, oats, pulao (0.80 - 0.88 g/ml)
    LENTILS_DALS = "lentils_dals"    # Cooked dals, sambar, rasam, dal makhani, chana (1.02 - 1.06 g/ml)
    CURRIES_GRAVIES = "curries_gravies" # Rich gravies, paneer butter masala, korma (0.98 - 1.05 g/ml)
    DRY_SABZI = "dry_sabzi"          # Dry cooked veggies, bhindi, aloo gobi, stir fry (0.65 - 0.75 g/ml)
    SALADS_GREENS = "salads_greens"  # Raw greens, cucumber, tomato kachumber, lettuce (0.25 - 0.35 g/ml)
    DAIRY_BEVERAGES = "dairy_beverages" # Curd, dahi, raita, yogurt, milk, lassi (1.02 - 1.06 g/ml)
    BREADS = "breads"                # Flatbreads (piece mass reference: roti ~40g, paratha ~75g)
    SNACKS_SWEETS = "snacks_sweets"  # Halwa, gulab jamun, samosa (0.90 - 1.10 g/ml)
    GENERIC = "generic"              # Unspecified aqueous/composite default (1.00 g/ml)


class DensityBounds(NamedTuple):
    nominal: float
    min_val: float
    max_val: float
    sd: float


# Certified laboratory culinary bulk densities (g/ml)
CERTIFIED_DENSITY_MAP: dict[DensityCategory, DensityBounds] = {
    DensityCategory.GRAINS: DensityBounds(nominal=0.84, min_val=0.80, max_val=0.88, sd=0.03),
    DensityCategory.LENTILS_DALS: DensityBounds(nominal=1.04, min_val=1.01, max_val=1.07, sd=0.02),
    DensityCategory.CURRIES_GRAVIES: DensityBounds(nominal=1.02, min_val=0.98, max_val=1.05, sd=0.02),
    DensityCategory.DRY_SABZI: DensityBounds(nominal=0.70, min_val=0.62, max_val=0.78, sd=0.05),
    DensityCategory.SALADS_GREENS: DensityBounds(nominal=0.28, min_val=0.22, max_val=0.35, sd=0.04),
    DensityCategory.DAIRY_BEVERAGES: DensityBounds(nominal=1.04, min_val=1.02, max_val=1.06, sd=0.01),
    DensityCategory.BREADS: DensityBounds(nominal=0.60, min_val=0.50, max_val=0.70, sd=0.06),
    DensityCategory.SNACKS_SWEETS: DensityBounds(nominal=0.95, min_val=0.85, max_val=1.10, sd=0.08),
    DensityCategory.GENERIC: DensityBounds(nominal=1.00, min_val=0.90, max_val=1.10, sd=0.05),
}

# Regex keywords mapping dishes to categories
CATEGORY_PATTERNS: list[tuple[DensityCategory, re.Pattern[str]]] = [
    (
        DensityCategory.SALADS_GREENS,
        re.compile(r"\b(salad|kachumber|cucumber|lettuce|cabbage|spinach|palak raw|greens|microgreens|sprouts)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.GRAINS,
        re.compile(r"\b(rice|chawal|biryani|pulao|pulav|khichdi|fried rice|jeera rice|oats|porridge|quinoa|millets|couscous)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.LENTILS_DALS,
        re.compile(r"\b(dal|daal|dhal|sambar|sambhar|rasam|rajma|chana|chole|lentil|moong|urad|toor|chickpea curry)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.DAIRY_BEVERAGES,
        re.compile(r"\b(curd|dahi|yogurt|raita|lassi|buttermilk|chaas|milk|chai|tea|coffee)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.BREADS,
        re.compile(r"\b(roti|chapati|phulka|naan|paratha|parotta|kulcha|puri|poori|bread|toast)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.DRY_SABZI,
        re.compile(r"\b(sabzi|sabji|bhindi|okra|aloo gobi|gobi masala|baingan|brinjal|eggplant|poriyal|stir fry|thoran|roast fry|dry fry)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.CURRIES_GRAVIES,
        re.compile(r"\b(curry|gravy|masala|paneer|korma|tikka masala|butter chicken|makhani|saag|kofta|kadhai)\b", re.IGNORECASE),
    ),
    (
        DensityCategory.SNACKS_SWEETS,
        re.compile(r"\b(halwa|gulab jamun|rasgulla|kheer|payasam|samosa|pakora|vada|bhajji|laddu|ladoo)\b", re.IGNORECASE),
    ),
]


class CulinaryDensityProfile(BaseModel):
    """Certified physical bulk density descriptor for volume-to-mass conversion."""

    model_config = ConfigDict(from_attributes=True)

    category: DensityCategory = Field(..., description="Culinary density category")
    nominal_density: float = Field(..., description="Nominal physical bulk density in g/ml")
    min_density: float = Field(..., description="Empirical minimum bulk density in g/ml")
    max_density: float = Field(..., description="Empirical maximum bulk density in g/ml")
    sd_density: float = Field(..., description="Standard deviation of bulk density in g/ml")
    matched_term: Optional[str] = Field(None, description="Matched keyword or regex token")


class CulinaryDensityService:
    """Service for resolving physical bulk density from dish semantics."""

    @classmethod
    def resolve_density(
        cls,
        dish_label: str,
        fallback: float = 1.00,
    ) -> CulinaryDensityProfile:
        """Resolve certified culinary bulk density profile from dish label string."""
        if not dish_label or not dish_label.strip():
            b = CERTIFIED_DENSITY_MAP[DensityCategory.GENERIC]
            return CulinaryDensityProfile(
                category=DensityCategory.GENERIC,
                nominal_density=round(fallback, 3),
                min_density=b.min_val,
                max_density=b.max_val,
                sd_density=b.sd,
                matched_term=None,
            )

        clean = dish_label.strip().lower()
        for cat, pattern in CATEGORY_PATTERNS:
            m = pattern.search(clean)
            if m:
                b = CERTIFIED_DENSITY_MAP[cat]
                return CulinaryDensityProfile(
                    category=cat,
                    nominal_density=b.nominal,
                    min_density=b.min_val,
                    max_density=b.max_val,
                    sd_density=b.sd,
                    matched_term=m.group(0),
                )

        # Fallback to Generic
        b = CERTIFIED_DENSITY_MAP[DensityCategory.GENERIC]
        return CulinaryDensityProfile(
            category=DensityCategory.GENERIC,
            nominal_density=round(fallback, 3),
            min_density=b.min_val,
            max_density=b.max_val,
            sd_density=b.sd,
            matched_term=None,
        )

    @classmethod
    def get_density_g_per_ml(
        cls,
        dish_label: str,
        fallback: float = 1.00,
    ) -> float:
        """Convenience method returning nominal bulk density in g/ml."""
        profile = cls.resolve_density(dish_label, fallback=fallback)
        return profile.nominal_density
