"""Vision services package for MessMate.

Modules:
- intrinsics: Camera pinhole optics and CIPA DC-008 normalization.
- reid: Tableware Re-ID with DINOv2 masked rim pooling and multimodal matching.
- depth: Metric depth scale pinning, specular inpainting, and volume integration.
"""

from app.services.vision.depth import DepthVolumeEngine, VolumeEstimate
from app.services.vision.intrinsics import (
    CameraIntrinsics,
    compute_focal_length_cipa,
    compute_metric_scale_anchor,
    extract_camera_intrinsics,
    project_3d_to_pixel,
    unproject_pixel_to_3d,
)
from app.services.vision.reid import (
    TablewareDescriptor,
    compute_tableware_similarity,
    extract_masked_rim,
    is_tableware_match,
    pool_masked_features,
)

__all__ = [
    "CameraIntrinsics",
    "compute_focal_length_cipa",
    "compute_metric_scale_anchor",
    "extract_camera_intrinsics",
    "project_3d_to_pixel",
    "unproject_pixel_to_3d",
    "TablewareDescriptor",
    "compute_tableware_similarity",
    "extract_masked_rim",
    "is_tableware_match",
    "pool_masked_features",
    "DepthVolumeEngine",
    "VolumeEstimate",
]
