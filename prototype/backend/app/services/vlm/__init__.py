"""VLM client package.

Adheres to MESSMATE_SPEC.md §9.
"""

from app.services.vlm.client import (
    MODEL_NAME,
    PROMPT_VERSION,
    VLMClient,
    VLMItem,
    VLMOutput,
)

__all__ = [
    "VLMClient",
    "VLMItem",
    "VLMOutput",
    "PROMPT_VERSION",
    "MODEL_NAME",
]
