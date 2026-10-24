"""Resolution service package.

Exposes ResolutionService and ResolvedDish.
"""

from app.services.resolution.engine import ResolutionService, ResolvedDish

__all__ = ["ResolutionService", "ResolvedDish"]
