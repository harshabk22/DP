"""Meal capture, processing, and retrieval endpoints conforming to SPEC §8."""

from datetime import date, datetime, timedelta, timezone
import logging
import math
from typing import Optional
import uuid
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.database import SessionLocal
from app.models.food import Food
from app.models.meal import Meal, MealItem
from app.models.question import DailyBudget
from app.models.user import User
from app.schemas.meal import (
    BudgetDetail,
    MealDetailResponse,
    MealItemDetail,
    MealTextRequest,
    MealTotalDetail,
    MealUploadResponse,
    QuestionEmbedded,
    QuestionOptionDetail,
)
from app.schemas.rollup import DayRollupResponse
from app.services.conformal.engine import ConformalCalibrationService
from app.services.nutrition.text_parser import TextMealParser
from app.services.resolution.engine import ResolutionService
from app.services.storage.media import MediaService
from app.tasks.meal_pipeline import (
    ARCHETYPE_MASS_G,
    ARCHETYPE_VOLUMES_ML,
    analyze_meal_task,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meals", tags=["meals"])


def _run_meal_analysis_bg(meal_id: uuid.UUID) -> None:
    """Worker entrypoint to execute meal pipeline in dedicated session with failure handling."""
    with SessionLocal() as db:
        try:
            analyze_meal_task(db, meal_id)
        except Exception as exc:
            db.rollback()
            logger.error(f"Async meal analysis failed for meal_id={meal_id}: {exc}", exc_info=True)
            meal = db.query(Meal).filter(Meal.id == meal_id).first()
            if meal:
                meal.status = "failed"
                db.commit()


@router.post(
    "",
    response_model=MealUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a meal image for asynchronous analysis (SPEC §8)",
)
@router.post(
    "/upload",
    response_model=MealUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a meal image for asynchronous analysis (SPEC §8 alias)",
)
@router.post(
    "/photo",
    response_model=MealUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a meal photo for asynchronous analysis (SPEC §8 alias)",
)
def upload_meal(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    x_client_meal_id: Optional[str] = Header(None, alias="X-Client-Meal-ID"),
    meal_id: Optional[str] = Form(None),
    client_meal_id: Optional[str] = Form(None),
    context: str = Form("personal_home"),
    meal_slot: Optional[str] = Form(None),
    captured_at: Optional[str] = Form(None),
    focal_length_mm: Optional[float] = Form(None),
    sensor_width_mm: Optional[float] = Form(None),
    sensor_height_mm: Optional[float] = Form(None),
    sync_run: bool = Query(False, description="Run analysis synchronously"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealUploadResponse:
    """Accept meal upload and trigger asynchronous analysis pipeline.

    Implements the decoupled 'Snap & Shove' pattern:
    - Accepts optional client-generated UUID via X-Client-Meal-ID or form payload.
    - Persists 'pending' state to database in <5 ms.
    - Enqueues asynchronous background analysis DAG (hit <40 ms synchronous SLA).
    - Returns HTTP 202 Accepted immediately.
    """
    now = datetime.now(timezone.utc)
    if captured_at:
        try:
            parsed_dt = datetime.fromisoformat(captured_at)
        except ValueError:
            parsed_dt = now
    else:
        parsed_dt = now

    # Resolve client-generated UUID meal_id (or generate if not provided)
    assigned_meal_id: Optional[uuid.UUID] = None
    if x_client_meal_id:
        try:
            assigned_meal_id = uuid.UUID(str(x_client_meal_id).strip())
        except (ValueError, AttributeError):
            assigned_meal_id = None

    if assigned_meal_id is None:
        for candidate_raw in (client_meal_id, meal_id):
            if candidate_raw:
                try:
                    assigned_meal_id = uuid.UUID(str(candidate_raw).strip())
                    break
                except (ValueError, AttributeError):
                    pass

    if assigned_meal_id is None:
        assigned_meal_id = uuid.uuid4()

    # Idempotent handling for client retries under cellular network jitter
    existing_meal = db.query(Meal).filter(Meal.id == assigned_meal_id).first()
    if existing_meal:
        return MealUploadResponse(
            meal_id=existing_meal.id,
            status=existing_meal.status,
            message="Meal already received.",
        )

    image_key = None
    if file is not None and getattr(file, "filename", None):
        content = file.file.read()
        if content:
            image_key, _, _ = MediaService.process_and_save_image(
                image_input=content,
                user_id=current_user.id,
                meal_id=assigned_meal_id,
            )

    context_map = {
        "home": "personal_home",
        "personal_home": "personal_home",
        "office": "personal_office",
        "work": "personal_office",
        "personal_office": "personal_office",
        "restaurant": "commercial",
        "away": "commercial",
        "commercial": "commercial",
    }
    normalized_context = context_map.get(str(context).lower().strip() if context else "personal_home", "personal_home")

    meal = Meal(
        id=assigned_meal_id,
        user_id=current_user.id,
        captured_at=parsed_dt,
        local_date=parsed_dt.date(),
        meal_slot=meal_slot or "lunch",
        source="photo",
        status="pending",
        context=normalized_context,
        image_key=image_key,
        image_purge_at=(now + timedelta(days=30)) if image_key else None,
    )
    db.add(meal)
    db.commit()
    db.refresh(meal)

    if sync_run:
        analyze_meal_task(db, meal.id)
    else:
        background_tasks.add_task(_run_meal_analysis_bg, meal.id)

    return MealUploadResponse(
        meal_id=meal.id,
        status="pending",
        message="Meal received and queued for analysis.",
    )


def _format_meal_detail(meal: Meal, current_user: User, db: Session) -> MealDetailResponse:
    """Helper to convert a Meal ORM object into the standard MealDetailResponse."""
    protein_total = 0.0
    fat_total = 0.0
    carb_total = 0.0
    fibre_total = 0.0

    items_detail: list[MealItemDetail] = []
    for item in meal.items:
        if item.food and item.grams_mean:
            ratio = float(item.grams_mean) / 100.0
            protein_total += float(item.food.protein_g or 0.0) * ratio
            fat_total += float(item.food.fat_g or 0.0) * ratio
            carb_total += float(item.food.carb_g or 0.0) * ratio
            fibre_total += float(item.food.fibre_g or 0.0) * ratio

        items_detail.append(
            MealItemDetail(
                id=item.id,
                dish_label=item.dish_label_raw,
                food_id=item.food_id,
                resolution_tier=item.resolution_tier,
                unit=item.unit,
                unit_count=item.unit_count,
                grams_mean=item.grams_mean,
                grams_sd=item.grams_sd,
                kcal_mean=item.kcal_mean,
                kcal_sd=item.kcal_sd,
                portion_source=item.portion_source,
                ask_value=item.ask_value,
                is_refill=item.is_refill,
                vlm_grams_raw=item.vlm_grams_raw,
            )
        )

    total_detail = None
    if meal.kcal_mean is not None:
        total_detail = MealTotalDetail(
            kcal_mean=round(float(meal.kcal_mean), 1),
            kcal_sd=round(float(meal.kcal_sd or 0.0), 1),
            kcal_p10=round(float(meal.kcal_p10 or 0.0), 1),
            kcal_p90=round(float(meal.kcal_p90 or 0.0), 1),
            protein_g=round(protein_total, 1),
            fat_g=round(fat_total, 1),
            carb_g=round(carb_total, 1),
            fibre_g=round(fibre_total, 1),
        )

    # Active question if allocator fired and question unanswered/unskipped (Invariant I9)
    question_embedded = None
    if meal.question and meal.question.answered_at is None and not meal.question.skipped:
        raw_opts = meal.question.options or []
        opts: list[QuestionOptionDetail] = []
        for opt in raw_opts:
            if isinstance(opt, dict):
                opts.append(QuestionOptionDetail(label=str(opt.get("label", opt)), value=opt.get("value", opt)))
            else:
                opts.append(QuestionOptionDetail(label=str(opt), value=opt))

        question_embedded = QuestionEmbedded(
            id=meal.question.id,
            type=meal.question.question_type,
            prompt=meal.question.prompt_text,
            options=opts,
        )

    # Budget status
    budget_row = (
        db.query(DailyBudget)
        .filter(DailyBudget.user_id == current_user.id, DailyBudget.local_date == meal.local_date)
        .first()
    )
    b_total = budget_row.budget_total if budget_row else 3
    b_used = budget_row.budget_used if budget_row else 0
    budget_detail = BudgetDetail(
        total=b_total,
        used=b_used,
        remaining=max(0, b_total - b_used),
    )

    return MealDetailResponse(
        meal_id=meal.id,
        status=meal.status,
        captured_at=meal.captured_at,
        context=meal.context,
        total=total_detail,
        items=items_detail,
        question=question_embedded,
        budget=budget_detail,
    )


@router.post(
    "/text",
    response_model=MealDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log a meal via natural language text or voice transcript (Human-First Assistant)",
)
def log_meal_text(
    payload: MealTextRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealDetailResponse:
    """Instantly parse and log a meal from natural language speech transcript or text."""
    now = datetime.now(timezone.utc)
    if payload.captured_at:
        try:
            parsed_dt = datetime.fromisoformat(payload.captured_at)
        except ValueError:
            parsed_dt = now
    else:
        parsed_dt = now

    parsed_items = TextMealParser.parse(payload.text)
    res_service = ResolutionService(db)

    meal_id = uuid.uuid4()
    db_items: list[MealItem] = []
    total_kcal_var = 0.0
    total_kcal_sum = 0.0

    for p in parsed_items:
        resolved = res_service.resolve(p.dish_label)
        food_id = resolved.food_id
        food = db.query(Food).filter(Food.id == food_id).first() if food_id else None

        # Determine unit grams
        if p.unit in ARCHETYPE_MASS_G:
            unit_g = ARCHETYPE_MASS_G[p.unit]
        elif p.unit in ARCHETYPE_VOLUMES_ML:
            density = float(food.density_g_per_ml) if (food and food.density_g_per_ml) else 1.0
            unit_g = ARCHETYPE_VOLUMES_ML[p.unit] * density
        else:
            unit_g = 100.0

        grams_mean = round(unit_g * float(p.unit_count), 1)
        grams_sd = round(grams_mean * 0.15, 1)

        kcal_per_g = (float(food.kcal_per_100g) / 100.0) if food else 1.0
        kcal_mean = round(grams_mean * kcal_per_g, 1)
        kcal_sd = round(grams_sd * kcal_per_g, 1)

        total_kcal_sum += kcal_mean
        total_kcal_var += (kcal_sd ** 2)

        meal_item = MealItem(
            id=uuid.uuid4(),
            meal_id=meal_id,
            food_id=food_id,
            dish_label_raw=p.dish_label,
            resolution_tier=resolved.resolution_tier,
            dish_uncertain=(resolved.confidence_score < 0.65),
            unit=p.unit,
            unit_count=float(p.unit_count),
            grams_mean=grams_mean,
            grams_sd=grams_sd,
            vlm_grams_raw=grams_mean,
            is_refill=False,
            kcal_mean=kcal_mean,
            kcal_sd=kcal_sd,
            portion_source="user",
            ask_value=0.0,
        )
        db_items.append(meal_item)

    total_sd = round(math.sqrt(total_kcal_var), 1)
    conf_bounds = ConformalCalibrationService.calibrate_meal_bounds(
        kcal_mean=total_kcal_sum,
        kcal_sd=total_sd,
    )
    p10 = conf_bounds.kcal_p10
    p90 = conf_bounds.kcal_p90

    meal = Meal(
        id=meal_id,
        user_id=current_user.id,
        captured_at=parsed_dt,
        local_date=parsed_dt.date(),
        meal_slot=payload.meal_slot or "lunch",
        source=payload.source if payload.source in ("photo", "voice", "manual", "text") else "text",
        status="analyzed",
        context=payload.context or "personal_home",
        kcal_mean=round(total_kcal_sum, 1),
        kcal_sd=total_sd,
        kcal_p10=p10,
        kcal_p90=p90,
        latency_ms=15,
    )
    db.add(meal)
    for it in db_items:
        db.add(it)

    db.commit()
    db.refresh(meal)

    return _format_meal_detail(meal, current_user, db)


@router.get(
    "/daily-rollup",
    response_model=DayRollupResponse,
    summary="Get daily nutrition rollup (legacy alias)",
    include_in_schema=False,
)
def get_day_rollup_legacy(
    date: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DayRollupResponse:
    """Legacy alias supporting /meals/daily-rollup?date=YYYY-MM-DD."""
    from app.api.v1.rollups import get_day_rollup
    day_str = date or "today"
    return get_day_rollup(day_str=day_str, current_user=current_user, db=db)


@router.get(
    "/{meal_id}",
    response_model=MealDetailResponse,
    summary="Retrieve meal analysis distribution and questions (SPEC §8)",
)
def get_meal(
    meal_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealDetailResponse:
    """Retrieve detailed analysis distribution and pending questions."""
    meal = (
        db.query(Meal)
        .filter(Meal.id == meal_id, Meal.user_id == current_user.id)
        .first()
    )
    if not meal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meal not found.",
        )

    return _format_meal_detail(meal, current_user, db)


@router.delete(
    "/{meal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Discard or delete a logged meal",
)
def delete_meal(
    meal_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Discard a logged meal."""
    meal = (
        db.query(Meal)
        .filter(Meal.id == meal_id, Meal.user_id == current_user.id)
        .first()
    )
    if not meal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meal not found.",
        )

    meal.status = "discarded"
    db.commit()
