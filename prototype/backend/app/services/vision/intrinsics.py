"""Camera pinhole optics and CIPA DC-008 diagonal FOV normalization.

Implements:
1. CIPA DC-008 diagonal field-of-view normalization:
   f_px = f_35mm * sqrt(width**2 + height**2) / 43.266615
2. Aspect-ratio-invariant focal length calculation.
3. Modern flagship fallback prior (f_35mm = 24.0 mm) replacing outdated 26mm priors.
4. Optical center computation (cx = width / 2.0, cy = height / 2.0).
5. Circular rim projective invariant metric scale anchoring:
   Z_anchor = (f_px * D_true) / d_major
6. 2D <-> 3D coordinate frame transformations.
"""

from dataclasses import dataclass
import io
import math
from typing import Any, Optional
from PIL import ExifTags, Image, ImageOps

# CIPA DC-008 diagonal normalization constant (mm)
# Standard 35mm film format: 36.0 mm x 24.0 mm -> sqrt(36^2 + 24^2) = 43.266615 mm
CIPA_35MM_DIAGONAL_MM: float = 43.266615

# Modern flagship wide lens equivalent focal length (iPhone 14-16, Samsung Galaxy S23-S24)
DEFAULT_FLAGSHIP_FOCAL_35MM: float = 24.0

# EXIF Tag IDs
EXIF_TAG_FOCAL_LENGTH_35MM = 41989  # FocalLengthIn35mmFilm
EXIF_TAG_FOCAL_LENGTH = 37386       # FocalLength


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic calibration parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    focal_35mm: float

    @property
    def f_px(self) -> float:
        """Scalar pixel focal length (assuming square pixels fx == fy)."""
        return self.fx

    @property
    def matrix(self) -> list[list[float]]:
        """3x3 Intrinsic camera calibration matrix K."""
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]


def compute_focal_length_cipa(
    width: int,
    height: int,
    focal_35mm: float = DEFAULT_FLAGSHIP_FOCAL_35MM,
) -> float:
    """Calculate focal length in pixels using CIPA DC-008 diagonal normalization.

    Formula:
        f_px = f_35mm * sqrt(width**2 + height**2) / 43.266615

    For a native 4:3 smartphone aspect ratio (W:H = 4:3, diagonal = 1.25 * W):
        f_px = (f_35mm / 34.6133) * W

    This eliminates the +8.17% volume inflation error caused by naive W / 36.0 normalization.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    if focal_35mm <= 0:
        focal_35mm = DEFAULT_FLAGSHIP_FOCAL_35MM

    diagonal_px = math.sqrt(float(width**2 + height**2))
    f_px = float(focal_35mm) * (diagonal_px / CIPA_35MM_DIAGONAL_MM)
    return f_px


def extract_camera_intrinsics(
    image_bytes: Optional[bytes] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    focal_35mm_override: Optional[float] = None,
) -> CameraIntrinsics:
    """Extract or construct camera intrinsics from image bytes or explicit dimensions.

    Uses EXIF metadata when available (FocalLengthIn35mmFilm tag 41989).
    Falls back to modern flagship prior (24.0 mm) if EXIF is absent or invalid.
    """
    detected_w = width
    detected_h = height
    detected_f35 = focal_35mm_override

    if image_bytes:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                img_w, img_h = img.size
                if not detected_w:
                    detected_w = img_w
                if not detected_h:
                    detected_h = img_h

                if detected_f35 is None:
                    exif = img.getexif()
                    if exif:
                        # Check primary EXIF
                        f35 = exif.get(EXIF_TAG_FOCAL_LENGTH_35MM)
                        if f35 is not None and float(f35) > 0:
                            detected_f35 = float(f35)
                        else:
                            # Check SubIFD if present
                            for ifd_id in (ExifTags.IFD.Exif,):
                                try:
                                    sub_ifd = exif.get_ifd(ifd_id)
                                    if sub_ifd and EXIF_TAG_FOCAL_LENGTH_35MM in sub_ifd:
                                        val = sub_ifd[EXIF_TAG_FOCAL_LENGTH_35MM]
                                        if val and float(val) > 0:
                                            detected_f35 = float(val)
                                            break
                                except Exception:
                                    pass
        except Exception:
            pass

    w = detected_w or 800
    h = detected_h or 800
    f35 = detected_f35 if (detected_f35 is not None and detected_f35 > 0) else DEFAULT_FLAGSHIP_FOCAL_35MM

    f_px = compute_focal_length_cipa(width=w, height=h, focal_35mm=f35)
    cx = float(w) / 2.0
    cy = float(h) / 2.0

    return CameraIntrinsics(
        fx=f_px,
        fy=f_px,
        cx=cx,
        cy=cy,
        width=w,
        height=h,
        focal_35mm=f35,
    )


def compute_metric_scale_anchor(
    f_px: float,
    d_true_m: float,
    d_major_px: float,
) -> float:
    """Compute anchor metric depth (Z_anchor in meters) using circular rim projective invariant.

    Formula:
        Z_anchor = (f_px * D_true) / d_major

    Where:
        - f_px: camera focal length in pixels
        - D_true: true physical diameter of the tableware container in meters
        - d_major: fitted rim ellipse major axis in pixels (unaffected by perspective slant)
    """
    if d_major_px <= 0:
        raise ValueError(f"Fitted rim major axis must be positive, got {d_major_px}")
    if f_px <= 0 or d_true_m <= 0:
        raise ValueError("f_px and d_true_m must be strictly positive")

    return (f_px * d_true_m) / d_major_px


def unproject_pixel_to_3d(
    u: float,
    v: float,
    z: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Unproject 2D pixel coordinate (u, v) with metric depth Z to 3D point (X, Y, Z)."""
    x = ((u - intrinsics.cx) * z) / intrinsics.fx
    y = ((v - intrinsics.cy) * z) / intrinsics.fy
    return (x, y, z)


def project_3d_to_pixel(
    x: float,
    y: float,
    z: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Project 3D point (X, Y, Z) to 2D image coordinates (u, v)."""
    if z <= 0:
        raise ValueError(f"Depth z must be positive for camera projection, got {z}")
    u = ((x * intrinsics.fx) / z) + intrinsics.cx
    v = ((y * intrinsics.fy) / z) + intrinsics.cy
    return (u, v)
