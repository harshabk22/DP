"""Unit tests for metric depth, specular bloom inpainting, and food volume integration."""

import math
import numpy as np
import pytest

from app.services.vision.depth import DepthVolumeEngine, VolumeEstimate
from app.services.vision.intrinsics import CameraIntrinsics


def test_specular_bloom_inpainting_stainless_steel():
    """Verify high-luminance specular bloom on stainless steel is inpainted."""
    h, w = 100, 100
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 120
    # Create specular reflection patch: V=255, S=0 (pure white glare)
    rgb[45:55, 45:55] = 255

    depth = np.ones((h, w), dtype=np.float32) * 0.40  # 40 cm
    depth[45:55, 45:55] = 0.0  # Sensor depth void at specular glare

    inpainted = DepthVolumeEngine.inpaint_specular_bloom(rgb, depth, v_thresh=245, s_thresh=0.18)
    assert not np.isnan(inpainted).any()
    # Inpainted glare void should be restored close to 0.40m
    assert inpainted[50, 50] > 0.35


def test_ransac_table_plane_fit():
    """Verify RANSAC fits horizontal table plane correctly."""
    np.random.seed(42)
    # Generate points on plane Z = 0.60 m with slight noise
    x = np.random.uniform(-0.2, 0.2, 200)
    y = np.random.uniform(-0.2, 0.2, 200)
    z = np.ones(200) * 0.60 + np.random.normal(0, 0.002, 200)
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)

    normal, d = DepthVolumeEngine.fit_table_plane_ransac(pts, n_iters=300)
    assert abs(normal[2]) > 0.90  # Normal should point predominantly in Z
    assert math.isclose(abs(d), 0.60, abs_tol=0.02)


def test_volume_by_subtraction_paraboloid_bowl():
    """Verify double surface integration matches theoretical volume of standard bowl.

    Standard bowl: Diameter = 14 cm (R=7cm), Depth = 6 cm.
    Parabolic volume V = 0.5 * pi * R^2 * H = 0.5 * pi * (7)^2 * 6 = 461.8 ml.
    When half-filled (thickness = 3cm): V ~ 230 ml.
    """
    w, h = 400, 400
    intrinsics = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=200.0, cy=200.0, width=w, height=h, focal_35mm=24.0
    )

    # Center bowl at image center at depth Z = 0.50 m
    yy, xx = np.ogrid[:h, :w]
    r_px = np.sqrt((xx - 200) ** 2 + (yy - 200) ** 2)

    # 14cm bowl at 0.5m with fx=500: radius in pixels = (0.07 / 0.50) * 500 = 70 px
    bowl_mask = r_px <= 70.0

    # Food surface at depth 0.50 m
    depth_map = np.ones((h, w), dtype=np.float32) * 0.50

    vol_est = DepthVolumeEngine.compute_volume_by_subtraction(
        depth_map=depth_map,
        food_mask=bowl_mask,
        intrinsics=intrinsics,
        rim_diameter_m=0.14,
        container_depth_m=0.06,
        shape_alpha=2.0,
        z_anchor_m=0.50,
    )

    assert vol_est.volume_ml > 200.0
    assert vol_est.volume_ml < 500.0
    assert vol_est.scale_factor_applied == 1.0
    assert vol_est.mean_thickness_cm > 1.5
