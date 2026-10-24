"""Background async tasks package."""

from app.tasks.meal_pipeline import analyze_meal_task

__all__ = ["analyze_meal_task"]
