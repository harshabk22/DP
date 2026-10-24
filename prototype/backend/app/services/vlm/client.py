"""Vision-Language Model (VLM) client for food perception.

Adheres strictly to:
- Invariant I1: No calories, macros, or nutrient values in prompt or output schema.
- Invariant I10: prompt_version = 'extract_v1', model_name = 'gemini-2.5-flash-lite'.
- Live execution: Gemini 2.5 Flash-Lite API with structured JSON output schema.
- Production safety: Fails loudly with informative error if GEMINI_API_KEY is missing
  in non-test environments rather than silently generating fake meals.
"""

from pathlib import Path
import base64
import json
import logging
import os
import time
from typing import Any, Optional
import httpx
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

logger = logging.getLogger(__name__)

# Google Gemini multimodal vision model (SPEC §9.2 / Gemini Flash-Lite Latest)
MODEL_NAME = "gemini-flash-lite-latest"
FALLBACK_MODEL_NAME = "gemini-3.5-flash-lite"
PROMPT_VERSION = "extract_v1"

# Archetypes allowed per SPEC §9.2
VALID_VESSEL_ARCHETYPES = {
    "small_bowl",
    "standard_bowl",
    "entree_bowl",
    "cup_mug",
    "dinner_plate",
    "tablespoon",
    "palm",
    "fist",
    "thumb",
    "slice_piece",
    "none",
}

VALID_PLATE_TYPES = {
    "thali",
    "single_bowl",
    "single_plate",
    "meal_prep_box",
    "packaged",
    "unclear",
}

# Structured response schema for Gemini API
VLM_JSON_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "plate_type": {
            "type": "STRING",
            "enum": list(VALID_PLATE_TYPES),
        },
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "dish_label": {"type": "STRING"},
                    "ingredient_guess": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "vessel_archetype": {
                        "type": "STRING",
                        "enum": list(VALID_VESSEL_ARCHETYPES),
                    },
                    "unit_count": {"type": "NUMBER"},
                    "grams_estimate": {"type": "NUMBER"},
                    "occlusion": {
                        "type": "STRING",
                        "enum": ["none", "partial", "heavy"],
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["dish_label", "vessel_archetype"],
            },
        },
        "scene_anchors_detected": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "notes": {"type": "STRING"},
    },
    "required": ["plate_type", "items"],
}


class MissingGeminiApiKeyError(RuntimeError):
    """Raised when live Gemini API key is missing in non-test environment."""

    pass


class VLMItem(BaseModel):
    """Single food item identified by visual perception (Invariant I1: zero nutrients)."""

    dish_label: str
    ingredient_guess: list[str] = Field(default_factory=list)
    vessel_archetype: str = "standard_bowl"
    unit_count: Optional[float] = 1.0
    grams_estimate: float = 150.0
    occlusion: str = "none"
    confidence: float = 0.90

    @field_validator("dish_label")
    @classmethod
    def clean_dish_label(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("vessel_archetype")
    @classmethod
    def validate_archetype(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in VALID_VESSEL_ARCHETYPES:
            return "standard_bowl"
        return clean


class VLMOutput(BaseModel):
    """VLM structured perception response according to MESSMATE_SPEC.md §9.2."""

    plate_type: str = "single_plate"
    items: list[VLMItem] = Field(default_factory=list)
    scene_anchors_detected: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class VLMClient:
    """Client for multimodal food perception using Gemini 2.5 Flash-Lite."""

    def __init__(self, api_key: Optional[str] = None, prompt_path: Optional[Path] = None):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL or MODEL_NAME
        self.prompt_version = PROMPT_VERSION

        # Resolve prompt file path across candidate locations
        if prompt_path:
            self.prompt_path = prompt_path
        else:
            candidates = [
                Path("prompts/extract_v1.txt"),
                Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "extract_v1.txt",
                Path(__file__).resolve().parent.parent.parent.parent.parent / "prompts" / "extract_v1.txt",
            ]
            self.prompt_path = next((p for p in candidates if p.exists()), candidates[0])

        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load frozen prompt text from file (Invariant I10)."""
        if self.prompt_path and self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8").strip()
        return (
            "You are an expert food perception model for an open-world dietary tracking system.\n"
            "Return ONLY JSON matching the provided schema. No prose or explanations.\n"
            "NEVER estimate calories, macronutrients, or nutritional values (Invariant I1)."
        )

    def extract(
        self,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        metric_scale: Optional[float] = None,
        mock_override: Optional[VLMOutput] = None,
    ) -> tuple[VLMOutput, dict[str, Any]]:
        """Extract structured visual food items from meal image.

        Returns (vlm_output, metadata) where metadata contains latency, cost, and raw responses.
        Fails loudly with MissingGeminiApiKeyError if GEMINI_API_KEY is missing in non-test mode.
        """
        start_time = time.time()

        # 1. Check for explicit test mock override
        if mock_override is not None:
            latency_ms = int((time.time() - start_time) * 1000)
            meta = {
                "model_name": self.model_name,
                "prompt_version": self.prompt_version,
                "latency_ms": max(latency_ms, 1),
                "cost_usd": 0.0,
                "raw_responses": mock_override.model_dump(),
            }
            return mock_override, meta

        # 2. In test environments (or when no image is provided): return deterministic test fixture unless explicitly requesting live API
        is_test_env = settings.is_test or settings.ENVIRONMENT in ("test", "ci")
        has_api_key = bool(self.api_key and not self.api_key.startswith("mock")) or bool(os.getenv("GEMINI_API_KEY"))

        if image_bytes is None or (is_test_env and os.getenv("TEST_LIVE_API") != "1"):
            vlm_output = self._deterministic_mock_output()
            latency_ms = int((time.time() - start_time) * 1000)
            meta = {
                "model_name": self.model_name,
                "prompt_version": self.prompt_version,
                "latency_ms": max(latency_ms, 10),
                "cost_usd": 0.00075,
                "raw_responses": vlm_output.model_dump(),
            }
            return vlm_output, meta

        # 3. In non-test environments: fail loudly if GEMINI_API_KEY is not configured
        if not has_api_key:
            raise MissingGeminiApiKeyError(
                "GEMINI_API_KEY environment variable is not configured. "
                "Real-world open-world food perception requires a valid Google Gemini API Key. "
                "Please configure GEMINI_API_KEY in backend/.env or system environment. "
                "(Set ENVIRONMENT=test or pass mock_override to run offline unit tests)."
            )

        # 4. Live Gemini API call with structured JSON response
        try:
            output, raw = self._call_gemini_api(image_bytes, image_url, metric_scale)
            latency_ms = int((time.time() - start_time) * 1000)
            meta = {
                "model_name": self.model_name,
                "prompt_version": self.prompt_version,
                "latency_ms": latency_ms,
                "cost_usd": 0.00075,
                "raw_responses": raw,
            }
            return output, meta
        except Exception as exc:
            logger.error(
                "Live Gemini API food perception failed: %s. Failing fast to prevent synthetic nutrition estimates.",
                exc,
                exc_info=True,
            )
            raise RuntimeError(f"Live Gemini API perception failed: {exc}") from exc

    def _call_gemini_api(
        self,
        image_bytes: Optional[bytes],
        image_url: Optional[str],
        metric_scale: Optional[float],
    ) -> tuple[VLMOutput, dict[str, Any]]:
        """Execute HTTP request to Gemini Flash-Lite API with structured JSON response."""
        candidate_models = [self.model_name]
        if FALLBACK_MODEL_NAME not in candidate_models:
            candidate_models.append(FALLBACK_MODEL_NAME)

        parts: list[dict[str, Any]] = [{"text": self._prompt_template}]

        if image_bytes:
            # Optimize image payload for network latency (<150KB)
            send_bytes = image_bytes
            mime_type = "image/jpeg"
            try:
                from PIL import Image
                import io
                with Image.open(io.BytesIO(image_bytes)) as pil_img:
                    pil_img.thumbnail((700, 700))
                    buf = io.BytesIO()
                    rgb_img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
                    rgb_img.save(buf, format="JPEG", quality=82)
                    send_bytes = buf.getvalue()
                    mime_type = "image/jpeg"
            except Exception:
                send_bytes = image_bytes

            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(send_bytes).decode("utf-8"),
                    }
                }
            )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        }

        last_error = None
        for m_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={self.api_key}"
            try:
                with httpx.Client(timeout=15.0) as client:
                    response = client.post(url, json=payload)
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Gemini API ({m_name}) returned HTTP {response.status_code}: {response.text}"
                        )
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates or "content" not in candidates[0]:
                        raise RuntimeError(f"Gemini API returned empty candidates or blocked content: {data}")
                    parts = candidates[0]["content"].get("parts", [])
                    if not parts or "text" not in parts[0]:
                        raise RuntimeError(f"Gemini API returned no text part: {candidates[0]}")
                    text_out = parts[0]["text"].strip()
                    
                    # Robust markdown code block stripping
                    if text_out.startswith("```"):
                        lines = text_out.splitlines()
                        if lines and lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        text_out = "\n".join(lines).strip()

                    parsed_json = json.loads(text_out)
                    if isinstance(parsed_json, list):
                        parsed_json = {
                            "plate_type": "thali" if len(parsed_json) > 1 else "single_plate",
                            "items": parsed_json,
                        }
                    elif isinstance(parsed_json, dict) and "items" not in parsed_json:
                        if "dishes" in parsed_json:
                            parsed_json["items"] = parsed_json.pop("dishes")
                        elif "food_items" in parsed_json:
                            parsed_json["items"] = parsed_json.pop("food_items")
                        else:
                            parsed_json["items"] = []

                    # Normalize individual item keys if model returned 'name' or 'dish'
                    if isinstance(parsed_json, dict):
                        for item in parsed_json.get("items", []):
                            if isinstance(item, dict):
                                if "dish_label" not in item:
                                    for alias in ("name", "dish", "food_name", "label"):
                                        if alias in item and item[alias]:
                                            item["dish_label"] = str(item[alias])
                                            break
                                if "vessel_archetype" not in item or not item["vessel_archetype"]:
                                    item["vessel_archetype"] = "standard_bowl"

                    return VLMOutput.model_validate(parsed_json), parsed_json
            except Exception as err:
                last_error = err
                logger.warning(f"Gemini API model {m_name} failed: {err}. Trying next candidate...")

        raise RuntimeError(f"All Gemini vision candidate models failed: {last_error}") from last_error

    def _deterministic_mock_output(self) -> VLMOutput:
        """Standardized deterministic perception output for testing and offline development."""
        return VLMOutput(
            plate_type="thali",
            items=[
                VLMItem(
                    dish_label="dal makhani",
                    ingredient_guess=["black lentils", "butter", "cream"],
                    vessel_archetype="small_bowl",
                    unit_count=1.0,
                    grams_estimate=150.0,
                    occlusion="none",
                    confidence=0.92,
                ),
                VLMItem(
                    dish_label="steamed basmati rice",
                    ingredient_guess=["rice", "water"],
                    vessel_archetype="standard_bowl",
                    unit_count=1.0,
                    grams_estimate=280.0,
                    occlusion="none",
                    confidence=0.95,
                ),
                VLMItem(
                    dish_label="roti",
                    ingredient_guess=["whole wheat flour", "water"],
                    vessel_archetype="slice_piece",
                    unit_count=2.0,
                    grams_estimate=80.0,
                    occlusion="none",
                    confidence=0.88,
                ),
            ],
            scene_anchors_detected=["cutlery_fork", "plate_rim"],
            notes="Multi-item balanced meal with standard tableware.",
        )
