"""fafir_registration.matching — LoFTR feature matching to an affine.

Turns the LoFTR model's dense correspondences into a partial-affine transform.
Matching runs on vessel *distance transforms* (``vdt_ir`` / ``vdt_faf``) rather
than raw masks, which gives LoFTR smooth, textured inputs to lock onto.

The LoFTR model itself lives in :mod:`fafir_registration.models.loftr`; its
availability (``loftr.LOFTR_AVAILABLE``) and handle (``loftr.loftr_matcher``)
are read here at call time so a lazy :func:`~fafir_registration.models.loftr.init_loftr`
in the pipeline is visible without re-importing.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import utils
from .models import loftr
from .preprocessing import close_faf_vessel, skeletonize


# ─────────────────────────────────────────────────────────────────────────────
# Vessel distance transforms (LoFTR inputs)
# ─────────────────────────────────────────────────────────────────────────────

def vdt(v, dil=1):
    sk = skeletonize(v)
    b  = (sk * 255).astype(np.uint8)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    d  = cv2.dilate(b, k, iterations=dil)
    dt = cv2.distanceTransform(d, cv2.DIST_L2, 5)
    if dt.max() > 0:
        dt = dt / dt.max()
    return dt.astype(np.float32)


def vdt_ir(v):
    return vdt((v), dil=3)


def vdt_faf(v):
    return vdt(close_faf_vessel(v), dil=3)


# ─────────────────────────────────────────────────────────────────────────────
# LoFTR matching
# ─────────────────────────────────────────────────────────────────────────────

def loftr_match(fv, mv_aligned, min_conf=0.4):
    if not loftr.LOFTR_AVAILABLE or loftr.loftr_matcher is None:
        return None, None
    import torch as th
    fdt = vdt_ir(fv).astype(np.float32)
    mdt = vdt_faf(mv_aligned).astype(np.float32)
    f_t = th.from_numpy(fdt).unsqueeze(0).unsqueeze(0).to(utils.DEVICE)
    m_t = th.from_numpy(mdt).unsqueeze(0).unsqueeze(0).to(utils.DEVICE)
    with th.no_grad():
        out = loftr.loftr_matcher({"image0": f_t, "image1": m_t})
    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    mask = conf > min_conf
    if mask.sum() < 8:
        print(f"  [LoFTR] only {int(mask.sum())} matches (conf>{min_conf})")
        return None, None
    print(f"  [LoFTR] {int(mask.sum())} confident matches")
    return kp1[mask].astype(np.float32), kp0[mask].astype(np.float32)


def loftr_to_affine(fv, mv_aligned, min_conf=0.4, params=None, bounds=None):
    src, dst = loftr_match(fv, mv_aligned, min_conf=min_conf)
    if src is None:
        return None, 0
    M, im = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                        ransacReprojThreshold=5.0,
                                        maxIters=5000, confidence=0.999)
    if M is None:
        return None, 0
    if not utils.sane(M, params, bounds):
        return None, 0
    return M.astype(np.float32), int(im.sum()) if im is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
# Image-intensity LoFTR (fallback when vessel masks are absent / pseudo)
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_loftr_input(img_rgb: np.ndarray) -> np.ndarray:
    """Return a modality-invariant, FOV-masked LoFTR representation.

    IR and FAF do not share intensity polarity, so direct CLAHE grayscale
    matching can lock onto modality-specific bright/dark regions. Gradient
    magnitude retains anatomical boundaries in both modalities while the FOV
    mask prevents the black image border from becoming a dominant feature.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    fov = utils.get_fov_mask(img_rgb) > 0

    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    # Robustly normalize only inside the retinal FOV. The weak CLAHE channel
    # preserves broad anatomical context while gradients drive matching.
    values = gradient[fov] if np.any(fov) else gradient.reshape(-1)
    lo, hi = np.percentile(values, (1.0, 99.0)) if values.size else (0.0, 0.0)
    grad_norm = np.clip((gradient - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    intensity = enhanced.astype(np.float32) / 255.0
    out = 0.85 * grad_norm + 0.15 * intensity
    out[~fov] = 0.0
    return out.astype(np.float32)


def loftr_match_image(fi_rgb: np.ndarray, mi_aligned_rgb: np.ndarray,
                      min_conf: float = 0.4,
                      min_matches: int = 8):
    """LoFTR on intensity images — used when vessel masks are missing/pseudo."""
    if not loftr.LOFTR_AVAILABLE or loftr.loftr_matcher is None:
        return None, None
    import torch as th
    f_t = (th.from_numpy(_img_to_loftr_input(fi_rgb))
             .unsqueeze(0).unsqueeze(0).to(utils.DEVICE))
    m_t = (th.from_numpy(_img_to_loftr_input(mi_aligned_rgb))
             .unsqueeze(0).unsqueeze(0).to(utils.DEVICE))
    with th.no_grad():
        out = loftr.loftr_matcher({"image0": f_t, "image1": m_t})
    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    mask = conf > min_conf
    if mask.sum() < int(min_matches):
        print(f"  [LoFTR-img] only {int(mask.sum())} matches (conf>{min_conf})")
        return None, None
    print(f"  [LoFTR-img] {int(mask.sum())} confident matches")
    return kp1[mask].astype(np.float32), kp0[mask].astype(np.float32)


def loftr_to_affine_image(fi_rgb: np.ndarray, mi_aligned_rgb: np.ndarray,
                          min_conf: float = 0.4,
                          min_matches: int = 8,
                          params=None, bounds=None):
    """Fit a partial-affine from image-intensity LoFTR correspondences."""
    src, dst = loftr_match_image(
        fi_rgb,
        mi_aligned_rgb,
        min_conf=min_conf,
        min_matches=min_matches,
    )
    if src is None:
        return None, 0
    M, im = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                        ransacReprojThreshold=3.0,
                                        maxIters=5000, confidence=0.999)
    if M is None:
        return None, 0
    if not utils.sane(M, params, bounds):
        return None, 0
    return M.astype(np.float32), int(im.sum()) if im is not None else 0