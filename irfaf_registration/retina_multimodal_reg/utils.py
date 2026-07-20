"""fafir_registration.utils — configuration, geometry, and affine sanity.

This is the lowest layer of the package: it holds all the mutable configuration
(FOV scale, scale-search window, sanity mode) plus the geometric primitives
(scale matrices, affine decomposition, warping) and the affine sanity gates.

Configuration is stored as module-level attributes and read by the rest of the
package via ``utils.NAME`` so that a single assignment (done in ``pipeline`` or
``registration``) is visible everywhere — this preserves the original script's
global-state behaviour after the split into modules.
"""

from __future__ import annotations

import re as _re

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Device / working size
# ─────────────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INF_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_H  = 224
TARGET_W  = 224

# ─────────────────────────────────────────────────────────────────────────────
# FOV scale — the ONLY source of the registration scale.
#   FAF = fixed (55°); IR = moving (30°). Moving IR is scaled by this factor.
#     55/30 = 1.833 ENLARGES (default; IR->FAF)
#     30/55 = 0.545 SHRINKS  (IR-moving-onto-FAF-fixed)
# The intended geometry for the current pipeline is the 55/30 enlargement.
# ─────────────────────────────────────────────────────────────────────────────
FOV_SCALE_RATIO = 55.0 / 30.0  # ≈ 1.833

# ── Narrow scale search (off by default; refined around the FOV seed) ─────────
SCALE_SEARCH_ENABLED   = False
SCALE_SEARCH_HALFWIDTH = 0.06
SCALE_SEARCH_STEP      = 0.01
SCALE_SEARCH_LO        = 1.60
SCALE_SEARCH_HI        = 1.95
# Require a real improvement before changing the seed scale; this avoids
# resolution-driven flips from tiny score differences after resizing.
SCALE_SEARCH_MARGIN    = 0.05

# ── Multi-scale LoFTR grid ───────────────────────────────────────────────────
MULTISCALE_LOFTR = False
MULTISCALE_MIN   = 1.30
MULTISCALE_MAX   = 1.95
MULTISCALE_STEP  = 0.05

# ── Objectives / final pick ──────────────────────────────────────────────────
OBJECTIVE            = "anatomical"
ASD_DIRECTED         = True
ASD_SELECTION_MARGIN = 0.01
# A vessel candidate below this score is treated as a failed/ambiguous vessel
# registration and is allowed to fall back to cross-modal image matching.
IMAGE_FALLBACK_MIN_ANATOMICAL = 0.45

# ── Adaptive per-pair sanity ─────────────────────────────────────────────────
ADAPTIVE_SANITY = False
SANITY_TOL      = 0.20
CURRENT_BOUNDS  = None          # set per-pair during registration
# Current processing row index (1-based). Set by the CLI before calling
# `register()` so other modules can inspect which CSV row is being handled.
CURRENT_ROW = None

# Cascade rollback tuning: by default, require only a very small anatomical
# gain to keep cascade refinements (this mirrors the earlier conservative
# behaviour). For targeted aggressive rollback on specific pairs, list their
# 1-based row indices in `CASCADE_AGGRESSIVE_ROWS`. Row 79 is enabled by
# default so the standard batch command visibly changes that pair.
CASCADE_IMG_DELTA = 0.08
CASCADE_ANAT_TOL = 0.02
CASCADE_ANAT_TOL_AGGRESSIVE = 0.06
CASCADE_AGGRESSIVE_ROWS = {79}

# ── Fixed affine sanity limits ───────────────────────────────────────────────
REJECT_ABS_ROT        = 120.0
REJECT_SCALE_MAX      = 4.0
REJECT_SCALE_MIN      = 0.1
REJECT_NEGATIVE_SCALE = True

FAF_CLOSE_KSIZE = 1
REJECT_SCALE_ANISO = 1.15
REJECT_SHEAR = 0.08
REFINEMENT_EPS = 0.01

# ── Image-readability flagging (intensity-based) ───────────────────────────
READABILITY_MIN_CONTRAST = 30.0
READABILITY_MIN_STD = 15.0
READABILITY_MIN_SHARP = 4.0
READABILITY_MIN_MEAN = 22.0
READABILITY_MAX_MEAN = 235.0
READABILITY_MIN_ENTROPY = 2.5
READABILITY_MAX_DARK_PCT = 0.70
READABILITY_MAX_BRIGHT_PCT = 0.20
READABILITY_MIN_FOV_COVERAGE = 0.35
READABILITY_MIN_KEYPOINTS = 50
READABILITY_MAX_NOISE = 6.0
READABILITY_MIN_FAILED_CHECKS = 1
# 1 = flag if either image fails, 2 = flag only if both fail.
READABILITY_MIN_FAILED_IMAGES = 2


# ─────────────────────────────────────────────────────────────────────────────
# Scale / FOV helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_scale_matrix(scale, cx, cy):
    return np.array([
        [scale, 0,     cx * (1 - scale)],
        [0,     scale, cy * (1 - scale)]
    ], dtype=np.float32)


def multiscale_grid(seed):
    grid = np.arange(MULTISCALE_MIN, MULTISCALE_MAX + 1e-9, MULTISCALE_STEP)
    vals = sorted({round(float(s), 3) for s in list(grid) + [float(seed)]})
    return vals


def parse_winner_scale(label, default):
    m = _re.search(r"@([0-9]+\.[0-9]+)", str(label))
    return round(float(m.group(1)), 3) if m else (round(float(default), 3)
                                                  if default is not None else None)


def scale_search_window(seed):
    if SCALE_SEARCH_LO is not None and SCALE_SEARCH_HI is not None:
        return float(SCALE_SEARCH_LO), float(SCALE_SEARCH_HI)
    return seed - SCALE_SEARCH_HALFWIDTH, seed + SCALE_SEARCH_HALFWIDTH


# ─────────────────────────────────────────────────────────────────────────────
# Warp utilities
# ─────────────────────────────────────────────────────────────────────────────

def warp_img(img_rgb, M, h, w):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def warp_mask(mv, M, h, w):
    out = cv2.warpAffine((mv * 255).astype(np.uint8), M, (w, h),
                         flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (out > 127).astype(np.float32)


def compose(M1, M2):
    def t(M): return np.vstack([M, [0, 0, 1]])
    return (t(M2) @ t(M1))[:2].astype(np.float32)


def get_fov_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return (mask > 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Affine decomposition & flagging
# ─────────────────────────────────────────────────────────────────────────────

def decompose_affine(M):
    tx, ty = float(M[0, 2]), float(M[1, 2])
    A = M[:2, :2].astype(np.float64)
    U, S, Vt = np.linalg.svd(A)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, -1] *= -1; S[-1] *= -1
    R = U @ Vt
    return dict(tx=tx, ty=ty,
                angle_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                sx=float(S[0]), sy=float(S[1]),
                shear=float((R.T @ A)[0, 1] / (float(S[0]) + 1e-8)))


def default_thresholds():
    return dict(max_translation=80.0, max_rotation=30.0,
                scale_lo=0.4, scale_hi=3.0, max_shear=0.25)


def flag_affine(decomp, thr):
    flags = []
    tx, ty = abs(decomp["tx"]), abs(decomp["ty"])
    if tx > thr["max_translation"] or ty > thr["max_translation"]:
        flags.append(f"large translation ({tx:.1f},{ty:.1f})px")
    ang = abs(decomp["angle_deg"])
    if ang > thr["max_rotation"]:
        flags.append(f"large rotation {ang:.2f}")
    for ax, s in [("sx", decomp["sx"]), ("sy", decomp["sy"])]:
        if not (thr["scale_lo"] <= s <= thr["scale_hi"]):
            flags.append(f"unusual scale {ax}={s:.4f}")
    if abs(decomp["shear"]) > thr["max_shear"]:
        flags.append(f"high shear {decomp['shear']:.4f}")
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Affine sanity gates
# ─────────────────────────────────────────────────────────────────────────────

def affine_is_sane(M, params=None):
    if M is None:
        return False
    A = M[:2, :2].astype(np.float64)
    U, S, Vt = np.linalg.svd(A)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        return False
    sx, sy = float(S[0]), float(S[1])
    R = U @ Vt
    ang = abs(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
    ang = min(ang, 360 - ang)
    if ang > REJECT_ABS_ROT:
        return False
    if REJECT_NEGATIVE_SCALE and (sx < 0 or sy < 0):
        return False
    if sx > REJECT_SCALE_MAX:
        return False
    if sy > REJECT_SCALE_MAX:
        return False
    if sx < REJECT_SCALE_MIN:
        return False
    if sy < REJECT_SCALE_MIN:
        return False
    # Reject strongly anisotropic scaling (sx/sy outside [1/REJECT_SCALE_ANISO, REJECT_SCALE_ANISO])
    if sy == 0 or sx / sy > REJECT_SCALE_ANISO or sy / sx > REJECT_SCALE_ANISO:
        return False
    # Reject excessive shear
    shear = float((R.T @ A)[0, 1] / (sx + 1e-8))
    if abs(shear) > REJECT_SHEAR:
        return False
    return True


def inframe_vessel_frac(mv, M, H, W):
    tot = float((mv > 0).sum())
    if tot < 1.0:
        return 1.0
    inside = float((warp_mask(mv, M, H, W) > 0).sum())
    ones = np.ones((H, W), np.float32)
    ones_in = float((warp_mask(ones, M, H, W) > 0).sum())
    keep = ones_in / float(H * W)
    if keep < 1e-6:
        return 0.0
    cov = inside / (tot * max(keep, 1e-6))
    return float(min(cov, 1.0))


def _seed_decomp_for_log(M_seed):
    d = decompose_affine(M_seed)
    return dict(
        seed_scale=round((d["sx"] + d["sy"]) / 2.0, 4),
        seed_angle=round(abs(d["angle_deg"]), 3),
        seed_tx=d["tx"],
        seed_ty=d["ty"],
    )


def seed_sanity_bounds(M_seed, fv=None, mv=None, tol=None):
    if tol is None:
        tol = SANITY_TOL
    if fv is None or mv is None:
        return dict(seed_coverage=None, coverage_floor=None,
                    _mv=None, _H=None, _W=None,
                    **_seed_decomp_for_log(M_seed))
    H, W = fv.shape
    seed_cov = inframe_vessel_frac(mv, M_seed, H, W)
    return dict(
        seed_coverage=round(seed_cov, 4),
        coverage_floor=max(seed_cov - tol, 0.0),
        _mv=mv, _H=H, _W=W,
        **_seed_decomp_for_log(M_seed),
    )


def affine_is_sane_adaptive(M, bounds):
    if M is None:
        return False
    if not affine_is_sane(M):
        return False
    if bounds is None:
        return True
    mv = bounds.get("_mv"); H = bounds.get("_H"); W = bounds.get("_W")
    floor = bounds.get("coverage_floor")
    if mv is None or floor is None:
        return True
    cov = inframe_vessel_frac(mv, M, H, W)
    return cov >= floor


def sane(M, params=None, bounds=None):
    """Unified sanity gate. Uses per-pair adaptive bounds when ADAPTIVE_SANITY
    is on and bounds are available (falling back to the module-global
    CURRENT_BOUNDS), otherwise the fixed-global affine_is_sane."""
    if bounds is None:
        bounds = CURRENT_BOUNDS
    if ADAPTIVE_SANITY and bounds is not None:
        return affine_is_sane_adaptive(M, bounds)
    return affine_is_sane(M, params)