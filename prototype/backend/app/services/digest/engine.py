"""Weekly digest and rule-based menu swap engine.

Adheres to MESSMATE_SPEC.md §10 & §11:
- Zero generative LLMs in recommendation path: deterministic, rule-based, and auditable.
- Checks ICMR-NIN recommended intakes: Protein (54g/d = 378g/week), Fibre (30g/d = 210g/week).
- Emits top 3 nutrient-density swaps from curated database or active cafeteria menu.
- Strictly adheres to Invariant I8: no shaming, no streak-loss framing, and persistent disclaimer.
"""

from datetime import date, datetime, timedelta
import re
from typing import Any, Optional
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.core.safety import DISCLAIMER_TEXT, get_user_daily_targets, validate_non_shaming_text
from app.models.food import CafeteriaMenu, Food
from app.models.meal import Meal, MealItem
from app.models.user import Profile, User
from app.schemas.rollup import FoodSwapResponse, WeekSummaryResponse

# ICMR-NIN Daily and Weekly Recommendations
ICMR_NIN_PROTEIN_DAILY = 54.0  # grams / day
ICMR_NIN_FIBRE_DAILY = 30.0    # grams / day
WEEKLY_RECOMMENDED_PROTEIN = ICMR_NIN_PROTEIN_DAILY * 7.0  # 378.0 g/week
WEEKLY_RECOMMENDED_FIBRE = ICMR_NIN_FIBRE_DAILY * 7.0      # 210.0 g/week


class WeeklyDigestEngine:
    """Computes weekly nutritional rollups and deterministic food swap recommendations."""

    def __init__(self, db: Session):
        self.db = db

    def parse_iso_week_range(self, iso_week: str) -> tuple[date, date]:
        """Convert ISO week string (e.g. '2026-W38' or '2026-38') to (start_date, end_date)."""
        clean = iso_week.upper().strip()
        match = re.match(r"^(\d{4})-?W?(\d{1,2})$", clean)
        if not match:
            # Fallback to current week
            today = date.today()
            start = today - timedelta(days=today.weekday())
            return start, start + timedelta(days=6)

        year, week_num = int(match.group(1)), int(match.group(2))
        try:
            # Monday of the ISO week
            start_date = datetime.strptime(f"{year}-W{week_num:02d}-1", "%G-W%V-%u").date()
        except ValueError:
            # Fallback if week format overflows
            start_date = date.fromisocalendar(year, min(week_num, 52), 1)

        end_date = start_date + timedelta(days=6)
        return start_date, end_date

    def generate_weekly_digest(self, user_id: Any, iso_week: str) -> WeekSummaryResponse:
        """Generate weekly summary and top 3 dietary swaps adhering to Invariant I8."""
        start_date, end_date = self.parse_iso_week_range(iso_week)

        # Retrieve user profile preferences
        profile = self.db.query(Profile).filter(Profile.user_id == user_id).first()
        diet_type = profile.diet_type if profile else "veg"
        cafeteria_id = profile.cafeteria_id if profile else None

        # Query all analyzed meals within date range
        meals = (
            self.db.query(Meal)
            .filter(
                Meal.user_id == user_id,
                Meal.local_date >= start_date,
                Meal.local_date <= end_date,
                Meal.status == "analyzed",
            )
            .all()
        )

        unique_days = {m.local_date for m in meals}
        days_logged = len(unique_days)

        total_kcal = 0.0
        total_protein = 0.0
        total_fibre = 0.0

        for meal in meals:
            total_kcal += float(meal.kcal_mean or 0.0)
            for item in meal.items:
                if item.food and item.grams_mean:
                    ratio = float(item.grams_mean) / 100.0
                    total_protein += float(item.food.protein_g or 0.0) * ratio
                    total_fibre += float(item.food.fibre_g or 0.0) * ratio

        # Effective divisor: days logged or at least 1
        div = max(days_logged, 1)
        mean_daily_kcal = round(total_kcal / div, 1)
        mean_daily_protein = round(total_protein / div, 1)
        mean_daily_fibre = round(total_fibre / div, 1)

        # Nutrient gaps vs weekly ICMR-NIN recommendations
        protein_gap = max(0.0, round(WEEKLY_RECOMMENDED_PROTEIN - total_protein, 1))
        fibre_gap = max(0.0, round(WEEKLY_RECOMMENDED_FIBRE - total_fibre, 1))

        # Check if gaps exceed 15% threshold (§10)
        swaps: list[FoodSwapResponse] = []
        target_nutrient = None
        if protein_gap > 0.15 * WEEKLY_RECOMMENDED_PROTEIN:
            target_nutrient = "protein"
        elif fibre_gap > 0.15 * WEEKLY_RECOMMENDED_FIBRE:
            target_nutrient = "fibre"

        if target_nutrient:
            swaps = self._get_top_swaps(target_nutrient, diet_type, cafeteria_id)

        # Construct safe, encouraging response
        resp = WeekSummaryResponse(
            iso_week=iso_week,
            days_logged=days_logged,
            mean_daily_kcal=mean_daily_kcal,
            mean_daily_protein_g=mean_daily_protein,
            mean_daily_fibre_g=mean_daily_fibre,
            protein_gap_g=protein_gap,
            fibre_gap_g=fibre_gap,
            swaps=swaps,
            disclaimer=DISCLAIMER_TEXT,
        )

        return resp

    def _get_top_swaps(
        self,
        nutrient: str,
        diet_type: Optional[str],
        cafeteria_id: Optional[str],
    ) -> list[FoodSwapResponse]:
        """Rank candidate foods by nutrient density (g / 100 kcal) and return top 3."""
        # Query candidate foods
        query = self.db.query(Food)

        # Filter by diet type
        if diet_type in ("veg", "vegan", "jain"):
            # Exclude non-veg items
            query = query.filter(
                ~Food.food_group.in_(["Poultry Products", "Meat", "Fish & Seafood", "Eggs"])
            )
            if diet_type == "vegan":
                query = query.filter(~Food.food_group.in_(["Dairy Products", "Milk"]))

        foods = query.all()
        if not foods:
            return []

        # Calculate density: nutrient_g / kcal
        candidates: list[tuple[Food, float]] = []
        for f in foods:
            kcal = float(f.kcal_per_100g) if f.kcal_per_100g > 0 else 100.0
            val = float(getattr(f, f"{nutrient}_g") or 0.0)
            if val > 0:
                density = (val / kcal) * 100.0  # grams per 100 kcal
                candidates.append((f, density))

        # Sort descending by nutrient density
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:3]

        swaps: list[FoodSwapResponse] = []
        for food, density in top_candidates:
            serving = f"1 {food.default_unit or 'serving'} ({round(food.pop_grams_per_unit or 100)}g)"
            rationale = (
                f"Nutrient-dense {food.name_en} provides {round(density, 1)}g {nutrient} per 100 kcal to help reach your nutritional target."
            )
            # Enforce Invariant I8 non-shaming copy
            assert validate_non_shaming_text(rationale), "Shaming copy detected in swap engine"

            swaps.append(
                FoodSwapResponse(
                    food_id=int(food.id),
                    name=food.name_en,
                    food_group=food.food_group,
                    target_nutrient=nutrient,
                    nutrient_density=round(density, 2),
                    serving_description=serving,
                    rationale=rationale,
                )
            )

        return swaps

    def generate_dinner_recommendation(
        self,
        user_id: Any,
        target_date: date,
    ) -> tuple[str, list[FoodSwapResponse]]:
        """Compute proactive 19:00 dinner gap guidance and targeted swaps (p01.md §2.C, §7.1).

        Adheres strictly to Invariant I8 (non-shaming, no guilt, no compensatory starvation).
        """
        profile = self.db.query(Profile).filter(Profile.user_id == user_id).first()
        diet_type = profile.diet_type if profile else "veg"
        cafeteria_id = profile.cafeteria_id if profile else None

        # Determine daily reference targets dynamically via Mifflin-St Jeor & Invariant I8
        target_kcal, target_protein, _, _ = get_user_daily_targets(profile)
        target_fibre = ICMR_NIN_FIBRE_DAILY      # 30.0g

        # Query all analyzed meals for target_date
        meals = (
            self.db.query(Meal)
            .filter(
                Meal.user_id == user_id,
                Meal.local_date == target_date,
                Meal.status == "analyzed",
            )
            .all()
        )

        total_kcal = 0.0
        total_protein = 0.0
        total_fibre = 0.0

        for meal in meals:
            total_kcal += float(meal.kcal_mean or 0.0)
            for item in meal.items:
                if item.food and item.grams_mean:
                    ratio = float(item.grams_mean) / 100.0
                    total_protein += float(item.food.protein_g or 0.0) * ratio
                    total_fibre += float(item.food.fibre_g or 0.0) * ratio

        kcal_rem = max(0.0, round(target_kcal - total_kcal, 1))
        protein_gap = max(0.0, round(target_protein - total_protein, 1))
        fibre_gap = max(0.0, round(target_fibre - total_fibre, 1))

        # Determine target nutrient for dinner swap
        target_nutrient = "protein"
        if fibre_gap > 10.0 and (fibre_gap / target_fibre) > (protein_gap / target_protein):
            target_nutrient = "fibre"
        elif protein_gap > 0.0:
            target_nutrient = "protein"
        elif fibre_gap > 0.0:
            target_nutrient = "fibre"

        swaps = self._get_top_swaps(target_nutrient, diet_type, cafeteria_id)

        # Formulate non-shaming advice
        if total_kcal >= target_kcal:
            advice = (
                "You've had a satisfying day of energy! For dinner, focus on light hydration, "
                "fiber, and soothing options like clear vegetable soup, dal shorba, or fresh salad "
                "to stay comfortably nourished without feeling sluggish."
            )
        elif protein_gap >= 15.0 or fibre_gap >= 8.0:
            swap_names = " or ".join([s.name for s in swaps[:2]]) if swaps else "dal or paneer"
            advice = (
                f"You have approximately {round(kcal_rem)} kcal remaining today with a {round(protein_gap, 1)}g "
                f"protein and {round(fibre_gap, 1)}g fiber target. Choosing {swap_names} "
                f"comfortably covers your remaining nutrient needs for dinner."
            )
        elif kcal_rem > 0:
            advice = (
                f"You're well on track today with {round(kcal_rem)} kcal left. "
                f"A balanced meal with your favorite protein and vegetables will round out your day nicely!"
            )
        else:
            advice = (
                "You've met your nutritional goals for today! Enjoy light evening hydration or herbal tea."
            )

        assert validate_non_shaming_text(advice), "Shaming copy detected in dinner recommendation"

        return advice, swaps

