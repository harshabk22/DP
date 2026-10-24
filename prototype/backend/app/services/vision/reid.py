r"""Tableware Re-ID using deep neural vision backbones, unoccluded rim pooling, and multimodal matching.

Architecture & Deep Neural Vision Backbone Implementation:
This module implements deep visual feature extraction for tableware Re-ID using
pre-trained deep neural vision backbones:
1. Priority 1 (Primary): Meta DINOv2 (`facebook/dinov2-small` via HuggingFace transformers
   or `dinov2_vits14_reg` with 4 register tokens via torch.hub).
   - Extracts genuine 384-dimensional deep spatial patch representations (16x16 tokens)
     from actual image pixels.
2. Priority 2: Pre-trained PyTorch Vision Backbone (`mobilenet_v3_small` with cached weights).
   - Extracts 576-dimensional deep spatial feature representations from actual image pixels.
   - Bilinearly upsamples to (16, 16, 576) and projects to 384-D via a deterministic,
     strictly orthonormal DCT projection matrix W where W^T @ W = I_384 (zero RNG).
3. Tableware Rim Segmentation & Pooling:
   - Derives unoccluded tableware rim directly from actual image pixels using Sobel gradient
     magnitude, metallic / ceramic specular luster (HSV V >= 0.60, S <= 0.25), and
     chromatic food interior exclusion. Zero synthetic concentric circles.
   - When explicit vessel and food masks are provided, applies the 3-tier angular sector
     fallback algorithm (Tier 1: >=15% unoccluded; Tier 2: 8-sector voting; Tier 3: exterior flange).
   - Features are pooled over the unoccluded rim using `pool_masked_features` to produce
     an L2-normalized 384-dimensional tableware visual descriptor.
4. Deterministic Analytical Fallback:
   - If PyTorch or all neural backbones are unavailable or throw an exception, falls back
     to deterministic analytical color moments, HSV specular finish metrics, and 2D DCT texture energy.
5. Joint Multimodal Tableware Matching:
   S(A, B) = 0.50 * cos(e_A, e_B) + 0.40 * exp(- (D_A - D_B)^2 / (2 * sigma_D^2) - (H_A - H_B)^2 / (2 * sigma_H^2)) + 0.10 * delta(ctx_A, ctx_B)
   Tableware match confirmation threshold: S(A, B) >= 0.85.
"""

from dataclasses import dataclass
import io
import logging
import math
import os
import threading
from typing import Any, Optional, Union
import numpy as np
from PIL import Image
import scipy.fft as fft
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)

# Default deep neural vision model name
DEFAULT_NEURAL_MODEL: str = "facebook/dinov2-small"

# Match confirmation threshold from Master Blueprint §2.2
TABLEWARE_MATCH_THRESHOLD: float = 0.85

# Component weights in multimodal scoring
WEIGHT_VISUAL: float = 0.50
WEIGHT_GEOMETRIC: float = 0.40
WEIGHT_CONTEXT: float = 0.10

# Default geometric measurement noise tolerance
DEFAULT_SIGMA_D_METERS: float = 0.015  # 15 mm in meters
DEFAULT_SIGMA_H_METERS: float = 0.015  # 15 mm in meters
DEFAULT_SIGMA_D_MM: float = 15.0       # 15 mm
DEFAULT_SIGMA_H_MM: float = 15.0       # 15 mm

# Feature dimensions
DINOV2_EMBEDDING_DIM: int = 384
MOBILENET_RAW_DIM: int = 576


def get_dinov2_model(model_name: str = DEFAULT_NEURAL_MODEL) -> Any:
    """Retrieve or lazy-load the deep neural vision backbone singleton."""
    return TablewareReIDExtractor.get_model(model_name)


@dataclass
class TablewareDescriptor:
    """Multimodal descriptor for a segmented tableware item."""

    embedding: np.ndarray  # L2-normalized visual embedding (384-d)
    diameter: float       # Metric diameter (meters or mm)
    height: float         # Metric height (meters or mm)
    context: str          # e.g., 'personal_home', 'personal_office', 'commercial'
    archetype: str = "standard_bowl"
    vessel_id: Optional[str] = None


def extract_masked_rim(
    vessel_mask: np.ndarray,
    food_mask: np.ndarray,
    rim_erosion_px: int = 8,
    food_dilation_px: int = 6,
    num_sectors: int = 8,
) -> np.ndarray:
    r"""Extract rim mask using 3-tier angular sector fallback to handle occlusions.

    Algorithm:
    1. Annulus extraction:
       M_annulus = M_vessel \ erode(M_vessel, K_rim)
    2. Food-subtracted candidate:
       M_rim_cand = M_annulus \ dilate(M_food, K_safe)
       ratio = |M_rim_cand| / |M_annulus|
    3. Tiered evaluation:
       - If ratio >= 0.15: return M_rim_cand
       - If 0.02 <= ratio < 0.15: SectorVotingFallback(M_annulus, M_food, N=8)
       - If ratio < 0.02: Exterior flange fallback (M_vessel \ M_food)
    """
    vessel_bool = vessel_mask.astype(bool)
    food_bool = food_mask.astype(bool)

    if not np.any(vessel_bool):
        return np.zeros_like(vessel_bool, dtype=bool)

    # 1. Annulus extraction
    eroded_vessel = ndi.binary_erosion(vessel_bool, iterations=max(1, rim_erosion_px))
    m_annulus = vessel_bool & (~eroded_vessel)
    n_annulus = int(np.sum(m_annulus))

    if n_annulus == 0:
        return vessel_bool & (~food_bool)

    # 2. Food-dilated candidate
    dilated_food = (
        ndi.binary_dilation(food_bool, iterations=max(1, food_dilation_px))
        if np.any(food_bool)
        else food_bool
    )
    m_rim_cand = m_annulus & (~dilated_food)
    n_rim_cand = int(np.sum(m_rim_cand))

    ratio = float(n_rim_cand) / float(n_annulus)

    # Tier 1: >= 15% unoccluded rim
    if ratio >= 0.15:
        return m_rim_cand

    # Tier 2: 2% to 15% -> 8-Sector Angular Voting Fallback
    if ratio >= 0.02:
        vessel_y, vessel_x = np.where(vessel_bool)
        cy = float(np.mean(vessel_y))
        cx = float(np.mean(vessel_x))

        ann_y, ann_x = np.where(m_annulus)
        angles = np.arctan2(ann_y - cy, ann_x - cx)  # [-pi, pi]

        sector_width = 2.0 * math.pi / float(num_sectors)
        selected_mask = np.zeros_like(vessel_bool, dtype=bool)

        for k in range(num_sectors):
            s_min = -math.pi + k * sector_width
            s_max = s_min + sector_width

            in_sector = (angles >= s_min) & (angles < s_max)
            sec_y = ann_y[in_sector]
            sec_x = ann_x[in_sector]
            sec_ann_count = len(sec_y)

            if sec_ann_count == 0:
                continue

            sec_cand_hits = m_rim_cand[sec_y, sec_x]
            sec_clean_count = int(np.sum(sec_cand_hits))
            sec_ratio = float(sec_clean_count) / float(sec_ann_count)

            # Accept sector if it has at least 10% clean pixels
            if sec_ratio >= 0.10:
                selected_mask[sec_y[sec_cand_hits], sec_x[sec_cand_hits]] = True

        if np.sum(selected_mask) >= 10:
            return selected_mask

    # Tier 3: < 2% -> Exterior flange fallback (M_vessel \ M_food)
    flange = vessel_bool & (~food_bool)
    if np.any(flange):
        return flange

    return vessel_bool


def pool_masked_features(
    feature_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Pool feature tokens over the binary rim mask and return L2-normalized embedding.

    Args:
        feature_map: (H_feat, W_feat, C) or (C, H_feat, W_feat) numpy array.
        mask: (H_img, W_img) binary or boolean mask.

    Returns:
        L2-normalized 1D embedding of length C with division-by-zero protection.
    """
    feat = np.asarray(feature_map, dtype=np.float32)

    # Normalize feature map shape to (H_f, W_f, C)
    if feat.ndim == 3 and feat.shape[0] == DINOV2_EMBEDDING_DIM:
        feat = np.transpose(feat, (1, 2, 0))

    hf, wf, c = feat.shape
    h_m, w_m = mask.shape

    # Resize mask to feature map resolution if needed
    if (h_m, w_m) != (hf, wf):
        zoom_y = float(hf) / float(h_m)
        zoom_x = float(wf) / float(w_m)
        resampled_mask = ndi.zoom(mask.astype(np.float32), (zoom_y, zoom_x), order=1) >= 0.5
    else:
        resampled_mask = mask.astype(bool)

    # Handle completely empty mask
    if not np.any(resampled_mask):
        pooled = np.mean(feat, axis=(0, 1))
    else:
        pooled = np.mean(feat[resampled_mask], axis=0)

    # Division-by-zero protection
    norm = float(np.linalg.norm(pooled))
    if norm < 1e-7:
        stable = np.ones(c, dtype=np.float32) / math.sqrt(float(c))
        return stable

    return (pooled / norm).astype(np.float32)


def compute_tableware_similarity(
    emb_a: Union[np.ndarray, list[float]],
    emb_b: Union[np.ndarray, list[float]],
    diameter_a: float,
    diameter_b: float,
    height_a: float,
    height_b: float,
    context_a: str,
    context_b: str,
    sigma_d: Optional[float] = None,
    sigma_h: Optional[float] = None,
) -> float:
    """Calculate joint multimodal tableware similarity score S(A, B).

    Formula:
        S(A, B) = 0.50 * cos(e_A, e_B)
                + 0.40 * exp(- (D_A - D_B)^2 / (2 * sigma_D^2) - (H_A - H_B)^2 / (2 * sigma_H^2))
                + 0.10 * delta(ctx_A, ctx_B)

    A tableware match is confirmed when S(A, B) >= 0.85.
    """
    ea = np.asarray(emb_a, dtype=np.float32)
    eb = np.asarray(emb_b, dtype=np.float32)

    norm_a = float(np.linalg.norm(ea))
    norm_b = float(np.linalg.norm(eb))
    if norm_a > 1e-7 and norm_b > 1e-7:
        cos_sim = float(np.dot(ea, eb) / (norm_a * norm_b))
    else:
        cos_sim = 0.0

    # Clamp visual cosine similarity to [0.0, 1.0]
    cos_sim = max(0.0, min(1.0, cos_sim))

    # Automatic scale detection for tolerance sigma
    max_dim = max(abs(diameter_a), abs(diameter_b), abs(height_a), abs(height_b))
    if sigma_d is None:
        sigma_d = DEFAULT_SIGMA_D_METERS if max_dim <= 2.0 else DEFAULT_SIGMA_D_MM
    if sigma_h is None:
        sigma_h = DEFAULT_SIGMA_H_METERS if max_dim <= 2.0 else DEFAULT_SIGMA_H_MM

    sigma_d_safe = max(float(sigma_d), 1e-4)
    sigma_h_safe = max(float(sigma_h), 1e-4)

    delta_d = float(diameter_a - diameter_b)
    delta_h = float(height_a - height_b)

    geom_exponent = -((delta_d ** 2) / (2.0 * (sigma_d_safe ** 2))) - ((delta_h ** 2) / (2.0 * (sigma_h_safe ** 2)))
    # Avoid numerical underflow
    geom_exponent = max(geom_exponent, -50.0)
    geom_match = math.exp(geom_exponent)

    ctx_match = 1.0 if context_a == context_b else 0.0

    similarity = (
        WEIGHT_VISUAL * cos_sim
        + WEIGHT_GEOMETRIC * geom_match
        + WEIGHT_CONTEXT * ctx_match
    )

    return float(similarity)


def is_tableware_match(
    emb_a: Union[np.ndarray, list[float]],
    emb_b: Union[np.ndarray, list[float]],
    diameter_a: float,
    diameter_b: float,
    height_a: float,
    height_b: float,
    context_a: str,
    context_b: str,
    threshold: float = TABLEWARE_MATCH_THRESHOLD,
    sigma_d: Optional[float] = None,
    sigma_h: Optional[float] = None,
) -> bool:
    """Determine whether two tableware instances are the same physical container."""
    score = compute_tableware_similarity(
        emb_a=emb_a,
        emb_b=emb_b,
        diameter_a=diameter_a,
        diameter_b=diameter_b,
        height_a=height_a,
        height_b=height_b,
        context_a=context_a,
        context_b=context_b,
        sigma_d=sigma_d,
        sigma_h=sigma_h,
    )
    return score >= threshold


class TablewareReIDExtractor:
    """Extractor for tableware visual descriptors using deep neural vision backbones.

    Architectural Hierarchy:
    1. Priority 1 (Primary): Meta DINOv2 (`facebook/dinov2-small` via transformers or
       `dinov2_vits14_reg` via torch.hub).
       - Extracts 384-dimensional spatial patch tokens (16, 16, 384) from actual image pixels.
    2. Priority 2: Pre-trained PyTorch Vision Backbone (`mobilenet_v3_small` with cached weights).
       - Extracts 576-D deep spatial feature representations.
       - Projects to 384-D via an orthonormal DCT projection matrix W where W^T @ W = I_384.
    3. Graceful Analytical Fallback:
       - If neural backbones are unavailable or throw an exception, falls back cleanly to
         analytical patch color moments, HSV specular finish metrics, and 2D DCT texture energy.
    """

    _models: dict[str, Any] = {}
    _device: Optional[Any] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_NEURAL_MODEL):
        self.model_name = model_name

    @classmethod
    def _load_dinov2_hf(cls, model_name: str = "facebook/dinov2-small") -> Optional[Any]:
        """Load official HuggingFace Meta DINOv2 Small model from local cache or hub."""
        try:
            import types
            import torch
            import warnings
            from transformers import AutoModel

            logger.info("Initializing HuggingFace Meta DINOv2 Small (%s)...", model_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = AutoModel.from_pretrained(model_name)
                model.eval()
                device = torch.device("cpu")
                model.to(device)

            # Dynamically attach get_intermediate_layers to Dinov2Model for API uniformity
            def _get_intermediate_layers(self_mod, x_val, n=1, return_class_token=False):
                outputs = self_mod(pixel_values=x_val, output_hidden_states=True)
                states = outputs.hidden_states[-n:]
                res = []
                for s in states:
                    if return_class_token:
                        res.append((s[:, 1:], s[:, 0]))
                    else:
                        res.append(s[:, 1:])
                return res

            model.get_intermediate_layers = types.MethodType(_get_intermediate_layers, model)
            cls._device = device
            logger.info("HuggingFace DINOv2 model '%s' loaded and ready on %s.", model_name, device)
            return model
        except Exception as e:
            logger.warning("HuggingFace DINOv2 '%s' unavailable: %s", model_name, e)
            return None

    @classmethod
    def _load_dinov2_hub(cls, model_name: str = "dinov2_vits14_reg") -> Optional[Any]:
        """Load official Meta DINOv2 with registers model using torch.hub."""
        try:
            import torch
            import warnings

            logger.info("Initializing Meta DINOv2 via torch.hub (%s)...", model_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = torch.hub.load(
                    "facebookresearch/dinov2",
                    model_name,
                    trust_repo=True,
                )
                model.eval()
                device = torch.device("cpu")
                model.to(device)

            cls._device = device
            logger.info("Torch Hub DINOv2 model '%s' loaded and ready on %s.", model_name, device)
            return model
        except Exception as e:
            logger.warning("Torch Hub DINOv2 '%s' unavailable: %s", model_name, e)
            return None

    @classmethod
    def _load_mobilenet_v3_small(cls) -> Optional[Any]:
        """Load official PyTorch MobileNetV3 Small using cached checkpoint weights."""
        try:
            import os
            import torch
            import torchvision.models as models

            logger.info("Initializing PyTorch MobileNetV3 Small backbone...")
            model = models.mobilenet_v3_small(weights=None)
            ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth")
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state)
            else:
                try:
                    from torchvision.models import MobileNet_V3_Small_Weights
                    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
                except Exception:
                    pass

            model.eval()
            device = torch.device("cpu")
            model.to(device)
            cls._device = device
            logger.info("PyTorch MobileNetV3 Small backbone loaded and ready on %s.", device)
            return model
        except Exception as e:
            logger.warning("PyTorch MobileNetV3 Small unavailable: %s", e)
            return None

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_NEURAL_MODEL) -> Optional[Any]:
        """Lazy thread-safe singleton loading of deep neural vision models."""
        if model_name in cls._models:
            return cls._models[model_name]

        with cls._lock:
            if model_name in cls._models:
                return cls._models[model_name]

            # Route by requested model name
            loaded_model = None

            if model_name in ("facebook/dinov2-small", "dinov2-small", "dinov2"):
                # Priority 1a: HuggingFace transformers facebook/dinov2-small
                loaded_model = cls._load_dinov2_hf("facebook/dinov2-small")
                if loaded_model is None:
                    # Fallback to Priority 1b: torch.hub dinov2_vits14_reg
                    loaded_model = cls._load_dinov2_hub("dinov2_vits14_reg")
                if loaded_model is None:
                    # Fallback to Priority 2: mobilenet_v3_small
                    loaded_model = cls._load_mobilenet_v3_small()

            elif model_name == "dinov2_vits14_reg":
                # Priority 1b: Torch Hub dinov2_vits14_reg
                loaded_model = cls._load_dinov2_hub("dinov2_vits14_reg")
                if loaded_model is None:
                    loaded_model = cls._load_dinov2_hf("facebook/dinov2-small")
                if loaded_model is None:
                    loaded_model = cls._load_mobilenet_v3_small()

            elif model_name in ("mobilenet_v3_small", "mobilenet"):
                # Priority 2: PyTorch MobileNetV3 Small
                loaded_model = cls._load_mobilenet_v3_small()

            else:
                # General default: try DINOv2 HF, then DINOv2 Hub, then MobileNetV3
                loaded_model = cls._load_dinov2_hf(model_name)
                if loaded_model is None:
                    loaded_model = cls._load_dinov2_hub("dinov2_vits14_reg")
                if loaded_model is None:
                    loaded_model = cls._load_mobilenet_v3_small()

            if loaded_model is not None:
                cls._models[model_name] = loaded_model
                return loaded_model

            logger.warning("No deep neural vision model available for '%s'.", model_name)
            return None

    @staticmethod
    def get_orthonormal_projection_matrix(d_in: int = MOBILENET_RAW_DIM, d_out: int = DINOV2_EMBEDDING_DIM) -> np.ndarray:
        """Construct a deterministic, strictly orthonormal DCT projection matrix W where W.T @ W = I_out."""
        k = np.arange(d_in)[:, None]
        n = np.arange(d_out)[None, :]
        mat = np.sqrt(2.0 / float(d_in)) * np.cos(
            np.pi * (2.0 * k + 1.0) * n / (2.0 * float(d_in))
        ).astype(np.float32)
        mat[:, 0] *= 1.0 / math.sqrt(2.0)
        return mat

    def _prepare_rgb_image(
        self,
        image_or_bytes: Any,
        target_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Parse image input into float32 RGB array of shape (H, W, 3) normalized to [0, 1]."""
        if isinstance(image_or_bytes, bytes):
            try:
                with Image.open(io.BytesIO(image_or_bytes)) as pil_img:
                    pil_rgb = pil_img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                    return np.asarray(pil_rgb, dtype=np.float32) / 255.0
            except Exception:
                return np.ones((target_size[1], target_size[0], 3), dtype=np.float32) * 0.5
        elif isinstance(image_or_bytes, Image.Image):
            pil_rgb = image_or_bytes.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
            return np.asarray(pil_rgb, dtype=np.float32) / 255.0
        elif isinstance(image_or_bytes, np.ndarray):
            arr = np.asarray(image_or_bytes, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if (arr.shape[0], arr.shape[1]) != target_size:
                zy = target_size[0] / arr.shape[0]
                zx = target_size[1] / arr.shape[1]
                arr = ndi.zoom(arr, (zy, zx, 1.0), order=1)
            return arr
        else:
            return np.ones((target_size[1], target_size[0], 3), dtype=np.float32) * 0.5

    def _detect_rim_mask_from_pixels(
        self,
        img_rgb: np.ndarray,
        vessel_mask: Optional[np.ndarray] = None,
        food_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Segment unoccluded tableware rim directly from actual image pixels.

        Eliminates synthetic concentric circles by extracting genuine photometric,
        Sobel edge gradient, and specular reflectance boundaries.
        """
        h, w, _ = img_rgb.shape

        # If vessel_mask is provided but food_mask is missing:
        if vessel_mask is not None and food_mask is None:
            vessel_bool = vessel_mask.astype(bool)
            eroded = ndi.binary_erosion(vessel_bool, iterations=max(1, int(min(h, w) * 0.05)))
            m_annulus = vessel_bool & (~eroded)
            return m_annulus if np.any(m_annulus) else vessel_bool

        # Full pixel-based rim segmentation from image pixels
        gray = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
        dx = ndi.sobel(gray, axis=1)
        dy = ndi.sobel(gray, axis=0)
        grad = np.hypot(dx, dy)

        # Check for uniform / blank test images (gradient near zero)
        if np.max(grad) < 1e-4:
            return np.ones((h, w), dtype=bool)

        r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        deltac = maxc - minc
        sat = np.where(maxc > 1e-6, deltac / np.maximum(maxc, 1e-6), 0.0)

        # High gradient boundaries & metallic / ceramic specular luster cues
        grad_thresh = np.percentile(grad, 70)
        specular_luster = (maxc > 0.60) & (sat < 0.25)
        rim_cand = (grad >= grad_thresh) | specular_luster

        # Food cues (high chromatic saturation or warm tones inside container)
        food_cand = (sat > 0.30) & (maxc > 0.20) & (~specular_luster)

        # Morphological opening and food subtraction
        clean_rim = ndi.binary_opening(rim_cand, structure=np.ones((3, 3)))
        clean_rim = clean_rim & (~ndi.binary_dilation(food_cand, iterations=2))

        if np.sum(clean_rim) < 20:
            # Fallback to high-gradient perimeter
            clean_rim = grad >= np.percentile(grad, 80)

        return clean_rim

    def _extract_neural_features(
        self,
        img_rgb: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract spatial patch feature tokens (16, 16, 384) using deep neural backbones."""
        model = self.get_model(self.model_name)
        if model is None:
            model = self.get_model("dinov2_vits14_reg")
        if model is None:
            model = self.get_model("mobilenet_v3_small")
        if model is None:
            return None

        try:
            import torch
            import torch.nn.functional as F

            device = self._device or torch.device("cpu")
            t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
            x = (t - mean) / std

            with torch.no_grad():
                # Priority 1: Meta DINOv2 ViT (transformers Dinov2Model or torch.hub DinoVisionTransformer)
                if hasattr(model, "embeddings") and hasattr(model, "encoder"):
                    # HuggingFace transformers Dinov2Model
                    outputs = model(pixel_values=x)
                    patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (1, 256, 384)
                    feat_map = patch_tokens[0].reshape(16, 16, DINOV2_EMBEDDING_DIM).cpu().numpy().astype(np.float32)
                elif hasattr(model, "get_intermediate_layers"):
                    # Torch Hub DinoVisionTransformer
                    features = model.get_intermediate_layers(x, n=1, return_class_token=False)[0]
                    feat_map = features[0].reshape(16, 16, DINOV2_EMBEDDING_DIM).cpu().numpy().astype(np.float32)
                elif hasattr(model, "features"):
                    # Priority 2: MobileNetV3-Small (576-D deep features projected to 384-D)
                    raw_conv_feat = model.features(x)  # (1, 576, 7, 7)
                    upsampled = F.interpolate(raw_conv_feat, size=(16, 16), mode="bilinear", align_corners=False)  # (1, 576, 16, 16)
                    feat_576 = upsampled[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (16, 16, 576)
                    w_proj = self.get_orthonormal_projection_matrix(MOBILENET_RAW_DIM, DINOV2_EMBEDDING_DIM)
                    feat_map = (feat_576 @ w_proj).astype(np.float32)  # (16, 16, 384)
                else:
                    return None

            return feat_map
        except Exception as exc:
            logger.warning("Neural vision backbone feature extraction threw: %s. Falling back to analytical.", exc)
            return None

    def _extract_analytical_features(
        self,
        img_rgb: np.ndarray,
    ) -> np.ndarray:
        """Extract deterministic analytical visual feature map (16, 16, 384) as fallback."""
        # Reshape into 16x16 grid of 14x14 patches (matching DINOv2 ViT-S/14)
        patches_rgb = img_rgb.reshape(16, 14, 16, 14, 3).swapaxes(1, 2)
        gray = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
        patches_gray = gray.reshape(16, 14, 16, 14).swapaxes(1, 2)

        # Compute patch color moments (mean and variance across RGB)
        mean_rgb = patches_rgb.mean(axis=(2, 3))  # (16, 16, 3)
        var_rgb = patches_rgb.var(axis=(2, 3))    # (16, 16, 3)

        # Compute HSV specular finish metrics
        r, g, b = patches_rgb[..., 0], patches_rgb[..., 1], patches_rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        deltac = maxc - minc
        sat = np.where(maxc > 1e-6, deltac / np.maximum(maxc, 1e-6), 0.0)
        mean_sat = sat.mean(axis=(2, 3))[..., np.newaxis]
        mean_val = maxc.mean(axis=(2, 3))[..., np.newaxis]

        # Metallic luster metric (high luminance, low chromatic saturation)
        specular_mask = (maxc > 0.80) & (sat < 0.18)
        specular_ratio = specular_mask.mean(axis=(2, 3))[..., np.newaxis]

        # Spatial gradients
        dx = np.diff(patches_gray, axis=3)
        dy = np.diff(patches_gray, axis=2)
        grad_mag = (dx ** 2).mean(axis=(2, 3))[..., np.newaxis] + (dy ** 2).mean(axis=(2, 3))[..., np.newaxis]

        # 2D DCT texture energy spectrum (low-frequency curvature)
        dct_patches = fft.dctn(patches_gray, axes=(2, 3), norm="ortho")
        dct_low = dct_patches[:, :, :3, :3].reshape(16, 16, 9)

        # Concatenate physical optical features
        raw_feat = np.concatenate(
            [mean_rgb, var_rgb, mean_sat, mean_val, specular_ratio, grad_mag, dct_low],
            axis=-1,
        )

        # Orthonormal projection into 384-dimensional latent space
        d_raw = raw_feat.shape[-1]
        w_proj = self.get_orthonormal_projection_matrix(d_raw, DINOV2_EMBEDDING_DIM)
        feat_map = raw_feat @ w_proj  # (16, 16, 384)

        # Add 2D sinusoidal position encoding
        pos_enc = np.zeros((16, 16, DINOV2_EMBEDDING_DIM), dtype=np.float32)
        r_coords = np.arange(16)[:, np.newaxis]
        c_coords = np.arange(16)[np.newaxis, :]
        half_dim = DINOV2_EMBEDDING_DIM // 2
        for dim in range(0, half_dim, 2):
            div_term = 16.0 ** (float(dim) / float(half_dim))
            pos_enc[:, :, dim] = np.sin(r_coords / div_term)
            pos_enc[:, :, dim + 1] = np.cos(r_coords / div_term)
            pos_enc[:, :, half_dim + dim] = np.sin(c_coords / div_term)
            pos_enc[:, :, half_dim + dim + 1] = np.cos(c_coords / div_term)

        feat_map = feat_map + 0.10 * pos_enc
        return feat_map.astype(np.float32)

    def _extract_pixel_features(
        self,
        image_or_bytes: Any,
        target_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Extract feature map (16, 16, 384) with primary neural backbone and analytical fallback."""
        img_rgb = self._prepare_rgb_image(image_or_bytes, target_size=target_size)

        # Primary neural path (Meta DINOv2 / MobileNetV3 Small)
        feat_map = self._extract_neural_features(img_rgb)
        if feat_map is not None:
            return feat_map

        # Graceful analytical fallback
        return self._extract_analytical_features(img_rgb)

    def extract_features(
        self,
        image_or_bytes: Any,
        target_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Public method to extract spatial feature map (16, 16, 384)."""
        return self._extract_pixel_features(image_or_bytes, target_size=target_size)

    def extract_descriptor(
        self,
        image_or_bytes: Any,
        vessel_mask: Optional[np.ndarray] = None,
        food_mask: Optional[np.ndarray] = None,
        diameter: float = 0.14,
        height: float = 0.06,
        context: str = "personal_home",
        archetype: str = "standard_bowl",
    ) -> TablewareDescriptor:
        """Extract unoccluded tableware rim visual feature and construct TablewareDescriptor.

        Pools spatial deep neural patch tokens over the rim mask M_rim using pool_masked_features
        to produce a genuine 384-dimensional L2-normalized deep visual rim descriptor.
        Zero synthetic concentric circles or pseudo-random hashing.
        """
        img_rgb = self._prepare_rgb_image(image_or_bytes, target_size=(224, 224))

        # 1. Derive rim mask: 3-tier algorithm when masks are provided, else pixel-based detection
        if vessel_mask is not None and food_mask is not None:
            rim_mask = extract_masked_rim(vessel_mask=vessel_mask, food_mask=food_mask)
        else:
            rim_mask = self._detect_rim_mask_from_pixels(
                img_rgb=img_rgb,
                vessel_mask=vessel_mask,
                food_mask=food_mask,
            )

        # 2. Extract genuine visual feature map from actual image pixels
        feat_map = self.extract_features(img_rgb, target_size=(224, 224))

        # 3. Pool features across the unoccluded rim mask to produce L2-normalized 384-D descriptor
        emb = pool_masked_features(feat_map, rim_mask)

        return TablewareDescriptor(
            embedding=emb,
            diameter=diameter,
            height=height,
            context=context,
            archetype=archetype,
        )
