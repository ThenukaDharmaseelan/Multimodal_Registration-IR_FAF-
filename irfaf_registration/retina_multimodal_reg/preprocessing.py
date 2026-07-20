"""irfaf_registration.preprocessing — image/mask preparation and density rules.

Intensity preprocessing (CLAHE + blur), IR vessel closing, skeletonization,
vessel-density measures, and the density-based heuristics (sparse detection,
quality tiering, segmentation-asymmetry / Dice-ceiling analysis).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import color, io

from . import utils


# ─────────────────────────────────────────────────────────────────────────────
# Intensity preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_rgb, clip_limit=3.0, tile_grid=(8, 8),
                     blur_ksize=5, blur_sigma=1.2,
                     auto_brighten=True, luma_target=105.0):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)

    # Mildly lift underexposed images for more stable feature matching.
    if auto_brighten:
        fov = (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) > 10)
        vals = l_eq[fov] if np.any(fov) else l_eq.reshape(-1)
        cur = float(np.mean(vals))
        if cur < luma_target:
            gamma = np.clip(cur / (luma_target + 1e-8), 0.70, 1.0)
            gain = np.clip(luma_target / (cur + 1e-8), 1.0, 1.12)
            l_norm = l_eq.astype(np.float32) / 255.0
            l_eq = np.clip((np.power(l_norm, gamma) * gain) * 255.0,
                           0, 255).astype(np.uint8)

    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    return cv2.GaussianBlur(enhanced, (ksize, ksize), blur_sigma)


def estimate_vessel_mask(img_rgb,
                         percentile=93.0,
                         min_pixels=32,
                         min_component_pixels=20,
                         min_vesselness=6.0):
    """Create a provisional vessel mask when no segmentation file exists.

    This is not a replacement for a trained vessel segmenter. It extracts
    dark, thin structures after CLAHE and a morphological black-hat filter so
    the normal vessel registration path can still be tried. The registration
    code retains image-based LoFTR as a fallback whenever this provisional
    vessel evidence is weak.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fov = utils.get_fov_mask(img_rgb) > 0
    if int(np.count_nonzero(fov)) < 64:
        return np.zeros(gray.shape, dtype=np.float32)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vesselness = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    values = vesselness[fov]
    threshold = float(np.percentile(values, percentile))
    # Guard against low-contrast scans where percentile-only thresholding
    # selects background texture/noise as pseudo vessels.
    threshold = max(threshold, float(min_vesselness))
    mask = ((vesselness >= threshold) & fov).astype(np.uint8)

    # Remove tiny speckles: pseudo vessels should be elongated structures,
    # not isolated dots from background noise.
    if int(np.count_nonzero(mask)) > 0 and int(min_component_pixels) > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_component_pixels):
                keep[labels == i] = 1
        mask = keep

    if int(np.count_nonzero(mask)) < int(min_pixels):
        return np.zeros(gray.shape, dtype=np.float32)
    return mask.astype(np.float32)


def close_faf_vessel(fv, ksize=None):
    if ksize is None:
        ksize = utils.FAF_CLOSE_KSIZE
    b = (fv * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return (cv2.morphologyEx(b, cv2.MORPH_CLOSE, k) > 0).astype(np.float32)


def skeletonize(v):
    b = (v * 255).astype(np.uint8)
    try:
        return (cv2.ximgproc.thinning(b) > 0).astype(np.float32)
    except Exception:
        return v


def vessel_density(v):
    return float(v.sum()) / (v.shape[0] * v.shape[1] + 1e-8)


def _vessel_component_stats(v,
                            min_component_pixels=24,
                            min_largest_component_pixels=48):
    """Summarize whether a vessel mask contains meaningful structure.

    Tiny isolated dots can have non-zero pixels yet still be unusable for
    registration. This helper measures connected-component size so callers can
    reject dot-only masks.
    """
    b = (np.asarray(v) > 0).astype(np.uint8)
    px = int(np.count_nonzero(b))
    if px == 0:
        return dict(px=0, largest=0, n_components=0, n_large=0, dot_like=True)

    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(b, connectivity=8)
    if num <= 1:
        return dict(px=px, largest=0, n_components=0, n_large=0, dot_like=True)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    largest = int(areas.max())
    n_components = int(len(areas))
    n_large = int(np.sum(areas >= int(min_component_pixels)))
    dot_like = (largest < int(min_largest_component_pixels)) or (n_large == 0)
    return dict(
        px=px,
        largest=largest,
        n_components=n_components,
        n_large=n_large,
        dot_like=bool(dot_like),
    )


def estimate_optic_disc_center(img_rgb,
                               min_area_frac=0.0008,
                               max_area_frac=0.08,
                               min_circularity=0.35):
    """Estimate an optic-disc center from a retinal image, without a disc mask.

    Both bright and dark compact extrema are considered because optic-disc
    polarity changes between IR and FAF acquisitions.  The return value is
    ``((x, y), confidence)`` or ``(None, 0.0)`` when no reliable candidate is
    found.  It is deliberately a soft landmark: callers should retain the
    normal vessel-based candidates and let anatomical scoring select the best.
    """
    if img_rgb is None:
        return None, 0.0
    img = np.asarray(img_rgb)
    if img.ndim != 3 or img.shape[2] < 3:
        return None, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fov = utils.get_fov_mask(img) > 0
    if not np.any(fov):
        return None, 0.0

    vals = gray[fov]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fov_area = int(np.count_nonzero(fov))
    if fov_area <= 0:
        return None, 0.0

    lo, hi = np.percentile(vals, (0.5, 99.5))
    if float(hi - lo) < 8.0:
        return None, 0.0

    best_center, best_score = None, 0.0
    for is_bright, threshold in ((True, hi), (False, lo)):
        cand = ((gray >= threshold) if is_bright else (gray <= threshold))
        cand = (cand & fov).astype(np.uint8)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
        num, labels, stats, centers = cv2.connectedComponentsWithStats(
            cand, connectivity=8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            area_frac = area / float(fov_area)
            if area_frac < min_area_frac or area_frac > max_area_frac:
                continue
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            peri = float(cv2.arcLength(cnts[0], True))
            if peri <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (peri * peri + 1e-8))
            if circularity < min_circularity:
                continue
            contrast = abs(float(gray[labels == i].mean()) - float(np.median(vals)))
            score = circularity * contrast / (float(hi - lo) + 1e-8)
            if score > best_score:
                best_center = tuple(float(v) for v in centers[i])
                best_score = float(score)

    return best_center, best_score


def has_optic_disc_candidate(img_rgb, **kwargs):
    """Return whether :func:`estimate_optic_disc_center` found a candidate."""
    center, _ = estimate_optic_disc_center(img_rgb, **kwargs)
    return center is not None


# ─────────────────────────────────────────────────────────────────────────────
# Density-based heuristics
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_params(fv, mv):
    H, W = fv.shape
    diag = float(np.sqrt(H ** 2 + W ** 2))
    fd = vessel_density(fv)
    md = vessel_density(mv)

    sparse_thresh = 0.01
    density_ratio = min(fd, md) / max(fd, md + 1e-8)
    sparse = (min(fd, md) < sparse_thresh) and (density_ratio < 0.30)

    if sparse:
        topo_weights = dict(w_cd=0.30, w_sd=0.25, w_bf=0.45)
    elif fd > 0.05 or md > 0.05:
        topo_weights = dict(w_cd=0.50, w_sd=0.40, w_bf=0.10)
    else:
        topo_weights = dict(w_cd=0.45, w_sd=0.35, w_bf=0.20)

    return dict(
        topo_weights=topo_weights,
        sparse=sparse,
        sparse_thresh=round(sparse_thresh, 6),
        density_ratio=round(density_ratio, 4),
        fv_density=fd,
        mv_density=md,
        img_diag=diag,
    )


def classify_pair_quality(fv, mv, fi=None, mi=None,
                          min_vessel_density=0.01,
                          good_vessel_density=0.03,
                          min_usable_vessel_pixels=24,
                          min_component_pixels=24,
                          min_largest_component_pixels=48):
    """Vessel-mask tier driven by vessel density (fv/mv are the vessel masks).

    This is a *segmentation* signal, not a flagging decision:

    - ``"good"``     — both masks have enough vessels; use vessel-based
      registration as normal.
    - ``"moderate"`` — one mask is sparse. This does NOT mean the pair is
      bad — the pipeline switches to image-based feature matching, and the
      caller should NOT flag the pair for this alone.
        - ``"bad"``      — at least one mask is effectively empty. Combined with
            a genuinely blank/unusable image this is folded into the "poor image
            quality" bucket upstream; it is the one tier that still contributes to
            image-based fallback.

    Optic-disc presence is not considered here — it was too unreliable a
    signal and was wrongly suppressing good vessel-based registrations for
    pairs that had perfectly usable vessel masks.
    """
    fv_stats = _vessel_component_stats(
        fv,
        min_component_pixels=min_component_pixels,
        min_largest_component_pixels=min_largest_component_pixels,
    )
    mv_stats = _vessel_component_stats(
        mv,
        min_component_pixels=min_component_pixels,
        min_largest_component_pixels=min_largest_component_pixels,
    )
    fv_pixels = int(fv_stats["px"])
    mv_pixels = int(mv_stats["px"])
    fd = float(vessel_density(fv))
    md = float(vessel_density(mv))
    lo = min(fd, md)
    reasons = []

    # Density alone must not label a visibly segmented vessel tree as
    # ``no_vessels``. A partial/small-FOV tree can easily occupy <1% of the
    # frame, but it remains meaningful for vessel-based registration.
    dot_like = bool(fv_stats["dot_like"] or mv_stats["dot_like"])
    if (min(fv_pixels, mv_pixels) < int(min_usable_vessel_pixels)) or dot_like:
        label = "bad"
        reasons.append(
            f"no_vessels(fv_px={fv_pixels},mv_px={mv_pixels},"
            f"fv_largest={int(fv_stats['largest'])},mv_largest={int(mv_stats['largest'])},"
            f"fv={fd:.4f},mv={md:.4f})"
        )
    elif min_vessel_density <= lo < good_vessel_density:
        label = "moderate"
        reasons.append(f"sparse_vessels(fv={fd:.4f},mv={md:.4f})")
    elif lo < min_vessel_density:
        label = "moderate"
        reasons.append(f"sparse_vessels(fv={fd:.4f},mv={md:.4f})")
    else:
        label = "good"
    return label, reasons


def _image_visibility_metrics(img_rgb):
    gray_u8 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32)
    fov = utils.get_fov_mask(img_rgb) > 0
    vals = gray[fov] if np.any(fov) else gray.reshape(-1)
    vals_u8 = gray_u8[fov] if np.any(fov) else gray_u8.reshape(-1)

    p5 = float(np.percentile(vals, 5.0))
    p95 = float(np.percentile(vals, 95.0))
    contrast = p95 - p5
    mean = float(np.mean(vals))
    std = float(np.std(vals))

    # Local contrast proxy via CLAHE-enhanced intensity spread.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    loc = clahe.apply(gray_u8)
    loc_vals = loc[fov] if np.any(fov) else loc.reshape(-1)
    local_contrast = float(np.std(loc_vals))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_fov = lap[fov] if np.any(fov) else lap.reshape(-1)
    sharp = float(np.mean(np.abs(lap_fov)))
    # Blur metric: variance of the Laplacian (standard focus-measure; low
    # values indicate a blurry image).
    blur_var = float(np.var(lap_fov))

    # Fast noise-level estimate (Immerkaer's method): convolve with a
    # Laplacian-of-Gaussian-like kernel that cancels out smooth structure,
    # leaving a signal proportional to additive noise sigma.
    noise_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    noise_resp = cv2.filter2D(gray, cv2.CV_32F, noise_kernel)
    noise_vals = noise_resp[fov] if np.any(fov) else noise_resp.reshape(-1)
    h_img, w_img = gray.shape[:2]
    noise_sigma = float(
        np.sqrt(np.pi / 2.0) / (6.0 * max(w_img - 2, 1) * max(h_img - 2, 1))
        * np.sum(np.abs(noise_vals))
    )

    hist, _ = np.histogram(vals_u8, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0

    dark_pct = float(np.mean(vals_u8 <= 5))
    bright_pct = float(np.mean(vals_u8 >= 250))
    fov_coverage = float(np.mean(fov)) if np.any(fov) else 0.0

    orb = cv2.ORB_create(nfeatures=500)
    kps = orb.detect(gray_u8, (fov.astype(np.uint8) * 255) if np.any(fov) else None)
    keypoints = int(len(kps) if kps is not None else 0)

    return dict(
        mean=mean,
        std=std,
        contrast=contrast,
        local_contrast=local_contrast,
        sharp=sharp,
        blur_var=blur_var,
        noise=noise_sigma,
        entropy=entropy,
        dark_pct=dark_pct,
        bright_pct=bright_pct,
        fov_coverage=fov_coverage,
        keypoints=keypoints,
    )


def classify_pair_readability(fi, mi,
                              min_contrast=None,
                              min_std=None,
                              min_sharp=None,
                              min_mean=None,
                              max_mean=None,
                              min_entropy=None,
                              max_dark_pct=None,
                              max_bright_pct=None,
                              min_fov_coverage=None,
                              min_keypoints=None,
                              max_noise=None,
                              min_failed_checks=None,
                              min_failed_images=None):
    """Automatic image-quality assessment, evaluated from image content only
    (no vessel/segmentation dependency).

    Runs each of IR/FAF through a battery of objective quality measures —
    blur (variance of Laplacian), noise estimate, global/local contrast,
    intensity range, entropy, dark/bright saturation, FOV coverage and
    ORB keypoint (feature) density — and flags an image as poor-quality when
    enough of these checks fail. The pair is flagged ``unreadable`` when at
    least ``min_failed_images`` of the two images are poor-quality, which
    lets low-quality cases be distinguished from genuine registration
    failures and routed to manual review instead of being scored as if
    registration were reliable.

    Any threshold left as ``None`` falls back to the corresponding
    ``utils.READABILITY_*`` default so this can run unattended with sensible
    defaults, while still being tunable per-call (or via the CLI flags that
    set those globals).

    Returns (unreadable: bool, reason: str).
    """
    min_contrast = utils.READABILITY_MIN_CONTRAST if min_contrast is None else min_contrast
    min_std = utils.READABILITY_MIN_STD if min_std is None else min_std
    min_sharp = utils.READABILITY_MIN_SHARP if min_sharp is None else min_sharp
    min_mean = utils.READABILITY_MIN_MEAN if min_mean is None else min_mean
    max_mean = utils.READABILITY_MAX_MEAN if max_mean is None else max_mean
    min_entropy = utils.READABILITY_MIN_ENTROPY if min_entropy is None else min_entropy
    max_dark_pct = utils.READABILITY_MAX_DARK_PCT if max_dark_pct is None else max_dark_pct
    max_bright_pct = utils.READABILITY_MAX_BRIGHT_PCT if max_bright_pct is None else max_bright_pct
    min_fov_coverage = utils.READABILITY_MIN_FOV_COVERAGE if min_fov_coverage is None else min_fov_coverage
    min_keypoints = utils.READABILITY_MIN_KEYPOINTS if min_keypoints is None else min_keypoints
    max_noise = getattr(utils, "READABILITY_MAX_NOISE", 6.0) if max_noise is None else max_noise
    min_failed_checks = (utils.READABILITY_MIN_FAILED_CHECKS if min_failed_checks is None
                         else min_failed_checks)
    min_failed_images = (utils.READABILITY_MIN_FAILED_IMAGES if min_failed_images is None
                         else min_failed_images)

    def _assess(image_or_path):
        if isinstance(image_or_path, (str, bytes)):
            img = io.imread(image_or_path)
        else:
            img = np.asarray(image_or_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        m = _image_visibility_metrics(img)

        fails = []
        if m["contrast"] < min_contrast:
            fails.append("contrast")
        if m["std"] < min_std:
            fails.append("std")
        if m["blur_var"] < min_sharp:
            fails.append("blur")
        if m["mean"] < min_mean or m["mean"] > max_mean:
            fails.append("mean")
        if m["entropy"] < min_entropy:
            fails.append("entropy")
        if m["dark_pct"] > max_dark_pct:
            fails.append("dark")
        if m["bright_pct"] > max_bright_pct:
            fails.append("bright")
        if m["fov_coverage"] < min_fov_coverage:
            fails.append("fov")
        if m["keypoints"] < min_keypoints:
            fails.append("keypoints")
        if m["noise"] > max_noise:
            fails.append("noise")

        bad = len(fails) >= int(min_failed_checks)
        detail = (
            f"mean={m['mean']:.1f},std={m['std']:.1f},c={m['contrast']:.1f},"
            f"lc={m['local_contrast']:.1f},blur={m['blur_var']:.2f},"
            f"noise={m['noise']:.2f},e={m['entropy']:.2f},k={m['keypoints']}"
        )
        return bad, fails, detail

    fi_bad, fi_fails, fi_detail = _assess(fi)
    mi_bad, mi_fails, mi_detail = _assess(mi)

    n_bad_images = int(fi_bad) + int(mi_bad)
    unreadable = n_bad_images >= int(min_failed_images)

    reasons = []
    if fi_bad:
        reasons.append(f"ir_poor_quality(fail={'+'.join(fi_fails)};{fi_detail})")
    if mi_bad:
        reasons.append(f"faf_poor_quality(fail={'+'.join(mi_fails)};{mi_detail})")
    if not unreadable:
        return False, ""
    return True, ";".join(reasons)


def segmentation_asymmetry(fv, mv, dsc_after=None, ratio_thresh=0.30):
    fpx = int((fv > 0).sum()); mpx = int((mv > 0).sum())
    fd = vessel_density(fv);   md = vessel_density(mv)
    ratio = min(fd, md) / max(fd, md + 1e-8)
    ceiling = 2.0 * min(fd, md) / (fd + md + 1e-8)
    out = dict(
        fv_vessel_px=fpx,
        mv_vessel_px=mpx,
        dice_ceiling=round(float(ceiling), 4),
        seg_limited=bool(ratio < ratio_thresh),
    )
    if dsc_after is not None and ceiling > 1e-6:
        out["dice_efficiency"] = round(min(float(dsc_after) / ceiling, 1.0), 4)
    return out
# preprocessing.py

"""Append these two functions to fafir_registration/preprocessing.py.

They power the vessel-segmentation flag used by pipeline.register(
flag_unsegmentable=..., flag_sparse=...) and the CLI. They rely on names
already defined in preprocessing.py: cv2, np, skeletonize.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Vessel-segmentation gate (dots/speckle & low-density detection)
# ─────────────────────────────────────────────────────────────────────────────

def assess_vessel_mask(
    v,
    min_usable_pixels=24,
    min_density=0.004,
    min_component_pixels=24,
    min_largest_extent_frac=0.15,   # largest comp bbox side / image side
    min_dominant_fraction=0.55,     # share of fg in components >= min_component_pixels
    max_fragmentation=0.05,         # n_components / fg_pixels  (noise -> high)
    min_skeleton_pixels=70,
):
    """Classify a vessel mask as 'ok' | 'sparse' | 'unsegmentable'.

    'unsegmentable' == dots/speckle/near-empty: no coherent vessel tree to
    register against. 'sparse' == a real but thin/partial tree (low density),
    still usable. Returns (label, reasons, stats).

    The signals separate a genuine (possibly partial) vessel tree from noise:

      density            foreground fraction of the frame
      largest_extent     max bbox side of the largest component / image side
                         (a real vessel spans the FOV; a blob does not)
      dominant_fraction  share of foreground pixels living in components
                         >= min_component_pixels (noise scatters into tiny bits)
      fragmentation      n_components / foreground_pixels (noise -> high)
      skeleton_pixels    total centerline length (dots skeletonize to ~nothing)
    """
    b = (np.asarray(v) > 0).astype(np.uint8)
    H, W = b.shape[:2]
    n_fg = int(b.sum())
    stats = dict(n_fg=n_fg, density=0.0, n_components=0, largest_extent=0.0,
                 dominant_fraction=0.0, fragmentation=0.0, skeleton_pixels=0)

    if n_fg < int(min_usable_pixels):
        return "unsegmentable", [f"near_empty(px={n_fg})"], stats

    density = n_fg / float(H * W)
    num, labels, cc, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    areas = cc[1:, cv2.CC_STAT_AREA].astype(np.int64)
    n_components = int(len(areas))
    imax = 1 + int(np.argmax(areas))
    ext = max(int(cc[imax, cv2.CC_STAT_WIDTH]), int(cc[imax, cv2.CC_STAT_HEIGHT]))
    largest_extent = ext / float(max(H, W))
    dominant_fraction = int(areas[areas >= int(min_component_pixels)].sum()) / float(n_fg)
    fragmentation = n_components / float(n_fg)
    skel_px = int((skeletonize(b.astype(np.float32)) > 0).sum())

    stats.update(density=round(density, 5), n_components=n_components,
                 largest_extent=round(float(largest_extent), 4),
                 dominant_fraction=round(float(dominant_fraction), 4),
                 fragmentation=round(float(fragmentation), 5),
                 skeleton_pixels=skel_px)

    # ── unsegmentable: speckle / scattered dots ──
    noise_flags = []
    if fragmentation > max_fragmentation:
        noise_flags.append(f"speckle(frag={fragmentation:.4f})")
    if dominant_fraction < min_dominant_fraction:
        noise_flags.append(f"fragmented(dom={dominant_fraction:.2f})")
    if largest_extent < min_largest_extent_frac:
        noise_flags.append(f"no_spanning_component(ext={largest_extent:.2f})")
    if skel_px < int(min_skeleton_pixels):
        noise_flags.append(f"short_skeleton({skel_px})")
    # two or more independent noise signals => not a real tree
    if len(noise_flags) >= 2:
        return "unsegmentable", noise_flags, stats

    # ── sparse: coherent but thin/partial ──
    if density < min_density or noise_flags:
        return "sparse", ([f"low_density({density:.4f})"] if density < min_density
                          else []) + noise_flags, stats

    return "ok", [], stats


def flag_vessel_pair(fv, mv, flag_sparse=False, **assess_kw):
    """Pair-level vessel-segmentation gate.

    Returns (flagged, verdict, reason, per_mask):
      verdict   the worse of the two masks ('ok' | 'sparse' | 'unsegmentable')
      flagged   True when either mask is 'unsegmentable' (dots/speckle), and
                also on 'sparse' when flag_sparse=True
      reason    per-mask reasons, prefixed 'IR:'/'FAF:'
      per_mask  {'ir': (label, stats), 'faf': (label, stats)}

    Extra keyword args (e.g. ``min_density=0.004``) are forwarded to
    ``assess_vessel_mask`` so the sparse/unsegmentable thresholds can be tuned
    per call.
    """
    order = {"ok": 0, "sparse": 1, "unsegmentable": 2}
    fl, fr, fs = assess_vessel_mask(fv, **assess_kw)
    ml, mr, ms = assess_vessel_mask(mv, **assess_kw)
    verdict = fl if order[fl] >= order[ml] else ml
    reasons = [f"IR:{r}" for r in fr] + [f"FAF:{r}" for r in mr]
    flagged = (verdict == "unsegmentable") or (flag_sparse and verdict == "sparse")
    return flagged, verdict, ";".join(reasons), dict(ir=(fl, fs), faf=(ml, ms))