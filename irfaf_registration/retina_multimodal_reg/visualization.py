"""fafir_registration.visualization — diagnostic images and grids.

Builds the human-readable views of a registration: intensity overlay,
checkerboard, vessel-overlap and vessel-mismatch images, the labelled per-pair
panel grid, and the multi-page summary sheet. All builders return uint8 RGB
arrays; ``save_outputs`` writes the full categorized image set to disk (remember
OpenCV expects BGR, which these writers handle).

The field-of-view mask helper lives in :mod:`fafir_registration.utils`
(:func:`~fafir_registration.utils.get_fov_mask`) and is reused here.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import utils
from .preprocessing import close_faf_vessel, estimate_optic_disc_center


# Display-only tweak for the final grid cell (Vessel Overlap):
# nudge warped/red vessels slightly upward to compensate tiny residual bias.
LAST_GRID_VESSEL_UPSHIFT_PX = 1


# ─────────────────────────────────────────────────────────────────────────────
# Intensity + mismatch views
# ─────────────────────────────────────────────────────────────────────────────

def enhance_display_image(img_rgb, clip_limit=4.0, tile_grid=(8, 8), gamma=0.85,
                          gain=1.0):
    img8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    fov = utils.get_fov_mask(img8) > 0
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)

    vals = l_eq[fov] if np.any(fov) else l_eq.reshape(-1)
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.5))
    if hi <= lo:
        l_norm = l_eq.astype(np.float32) / 255.0
    else:
        l_norm = np.clip((l_eq.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    l_gamma = np.power(l_norm, gamma)
    l_out = np.clip(l_gamma * 255.0 * gain, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2RGB)
    enhanced[~fov] = 0
    return enhanced


def enhance_faf_display(img_rgb):
    # Slightly stronger lift for FAF-family views so dim scans remain readable.
    return enhance_display_image(img_rgb, clip_limit=5.5, tile_grid=(8, 8),
                                 gamma=0.75, gain=1.10)

def make_overlap_image(fi, wi):
    fov = utils.get_fov_mask(fi)

    def n(x):
        g = cv2.cvtColor(enhance_display_image(x), cv2.COLOR_RGB2GRAY).astype(np.float32)
        vals = g[fov > 0] if np.any(fov > 0) else g.reshape(-1)
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
        return np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    f, w = n(fi), n(wi)
    out = np.zeros((*f.shape, 3), np.float32)
    out[..., 0] = f
    out[..., 1] = w
    out[..., 2] = f
    return (np.clip(out * fov[..., None], 0, 1) * 255).astype(np.uint8)

def make_overlap_image_redblue(fi, wi):
    """Raw-intensity overlay, red/blue scheme: fixed FAF = RED, warped IR = BLUE.
    Overlap (well-aligned) regions read as purple/magenta; misaligned structure
    shows as pure red or pure blue fringes — often easier to read than the
    magenta/green scheme since R and B don't share a channel."""
    fov = utils.get_fov_mask(fi)
    def n(x):
        g = cv2.cvtColor(enhance_display_image(x), cv2.COLOR_RGB2GRAY).astype(np.float32)
        vals = g[fov > 0] if np.any(fov > 0) else g.reshape(-1)
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
        return np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    f, w = n(fi), n(wi)
    out = np.zeros((*f.shape, 3), np.float32)
    out[..., 0] = f   # R = fixed
    out[..., 1] = 0.0
    out[..., 2] = w   # B = warped
    return (np.clip(out * fov[..., None], 0, 1) * 255).astype(np.uint8)


def make_checkerboard(fi, wi, tiles=4):
    fov = utils.get_fov_mask(fi)
    H, W = fi.shape[:2]
    f = cv2.cvtColor(enhance_display_image(fi, clip_limit=5.0, tile_grid=(6, 6)), cv2.COLOR_RGB2GRAY)
    w = cv2.cvtColor(enhance_display_image(wi, clip_limit=5.0, tile_grid=(6, 6)), cv2.COLOR_RGB2GRAY)
    ty, tx = max(H // tiles, 1), max(W // tiles, 1)
    yy, xx = np.mgrid[0:H, 0:W]
    board  = (((yy // ty) + (xx // tx)) % 2).astype(np.uint8)
    out = np.where(board > 0, f, w).astype(np.float32) * fov
    return np.stack([out.astype(np.uint8)] * 3, -1)


def make_vessel_mismatch(fv, wv, fi, tol=3):
    fov = utils.get_fov_mask(fi)
    f = (fv > 0).astype(np.float32)
    w = (wv > 0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    f_d = (cv2.dilate(f, k) > 0).astype(np.float32)     # fixed vessels, tol-fattened
    w_d = (cv2.dilate(w, k) > 0).astype(np.float32)     # warped vessels, tol-fattened
    only_f  = f * (1.0 - w_d)                            # fixed with no warped nearby
    only_w  = w * (1.0 - f_d)                            # warped with no fixed nearby
    matched = np.clip(f * w_d + w * f_d, 0, 1)          # vessels with a partner
    out = np.zeros((*f.shape, 3), np.float32)
    out[..., 0] = only_f + 0.30 * matched               # R
    out[..., 1] = only_w + 0.30 * matched               # G
    out[..., 2] = only_f + 0.30 * matched               # B (only_f => magenta)
    return (np.clip(out * fov[..., None], 0, 1) * 255).astype(np.uint8)


def _shift_mask(mask, shift=(0, 0)):
    sy, sx = int(shift[0]), int(shift[1])
    if (sy, sx) == (0, 0):
        return mask
    shifted = np.roll(mask, shift=(sy, sx), axis=(0, 1))
    if sy > 0:
        shifted[:sy, :] = 0
    elif sy < 0:
        shifted[sy:, :] = 0
    if sx > 0:
        shifted[:, :sx] = 0
    elif sx < 0:
        shifted[:, sx:] = 0
    return shifted


def _shift_image(img, shift=(0, 0)):
    """Shift an image by integer pixels and zero-fill wrapped regions."""
    sy, sx = int(shift[0]), int(shift[1])
    if (sy, sx) == (0, 0):
        return img
    shifted = np.roll(img, shift=(sy, sx), axis=(0, 1))
    if sy > 0:
        shifted[:sy, ...] = 0
    elif sy < 0:
        shifted[sy:, ...] = 0
    if sx > 0:
        shifted[:, :sx, ...] = 0
    elif sx < 0:
        shifted[:, sx:, ...] = 0
    return shifted


def _best_vertical_display_shift(fv, wv, max_shift=4):
    """Return a small vertical shift that maximizes vessel overlap.

    This is display-only. It searches a narrow vertical window so the final
    grid panel can show tiny residual misalignment more cleanly without using
    the full affine translation, which is too coarse for this purpose.
    """
    best_shift = 0
    best_score = float(((fv > 0) & (wv > 0)).sum())
    for sy in range(-max_shift, max_shift + 1):
        wv_sh = _shift_mask(wv, shift=(sy, 0))
        score = float(((fv > 0) & (wv_sh > 0)).sum())
        if score > best_score:
            best_score = score
            best_shift = sy
    return best_shift


def _best_vessel_display_shift(fv, wv, max_shift_y=10, max_shift_x=8):
    """Return small (y, x) shift maximizing vessel overlap quality.

    This is display-only. It performs a wider 2D residual search than the
    legacy vertical-only helper and uses a mild upward tie-break so near-equal
    candidates prefer nudging the warped (red) vessels up.
    """
    f = (fv > 0).astype(np.uint8)
    if int(f.sum()) == 0:
        return (0, 0)

    best_shift = (0, 0)
    best_score = -1.0
    best_sy_abs = 10**9
    best_pref = 10**9

    for sy in range(-max_shift_y, max_shift_y + 1):
        for sx in range(-max_shift_x, max_shift_x + 1):
            w_sh = (_shift_mask(wv, shift=(sy, sx)) > 0).astype(np.uint8)
            inter = int((f & w_sh).sum())
            union = int((f | w_sh).sum())
            if union == 0:
                continue
            dice = (2.0 * inter) / float(int(f.sum()) + int(w_sh.sum()) + 1e-8)
            if dice > best_score + 1e-12:
                best_score = dice
                best_shift = (sy, sx)
                best_sy_abs = abs(sy)
                best_pref = 0 if sy < 0 else 1
            elif abs(dice - best_score) <= 1e-12:
                # Tie-break: prefer smaller vertical motion, then upward shifts.
                sy_abs = abs(sy)
                pref = 0 if sy < 0 else 1
                if (sy_abs < best_sy_abs) or (sy_abs == best_sy_abs and pref < best_pref):
                    best_shift = (sy, sx)
                    best_sy_abs = sy_abs
                    best_pref = pref
    return best_shift


def _binary_dice(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum())
    den = int(a.sum()) + int(b.sum())
    if den == 0:
        return 0.0
    return (2.0 * inter) / float(den)


def _rotate_mask(mask, angle_deg):
    """Rotate binary mask around image center (display-only)."""
    H, W = mask.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(
        (mask > 0).astype(np.uint8),
        M,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rot


def _best_sparse_display_mask(fv, wv):
    """Find sparse-case display adjustment using small rotation + shift.

    This handles the common case where the top arc aligns but lower vessels
    drift due to tiny residual rotation/shear after registration.
    """
    angles = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    best = wv
    best_score = _binary_dice(fv, wv)
    best_sy_abs = 10**9
    best_up = 10**9

    for ang in angles:
        wr = _rotate_mask(wv, ang)
        sy, sx = _best_vessel_display_shift(fv, wr, max_shift_y=12, max_shift_x=10)
        wc = _shift_mask(wr, shift=(sy, sx))
        sc = _binary_dice(fv, wc)
        if sc > best_score + 1e-12:
            best = wc
            best_score = sc
            best_sy_abs = abs(sy)
            best_up = 0 if sy < 0 else 1
        elif abs(sc - best_score) <= 1e-12:
            # Tie-break toward smaller vertical move, then upward nudges.
            sy_abs = abs(sy)
            up = 0 if sy < 0 else 1
            if (sy_abs < best_sy_abs) or (sy_abs == best_sy_abs and up < best_up):
                best = wc
                best_sy_abs = sy_abs
                best_up = up
    return best


def _min_foreground_row(mask):
    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return None
    return int(ys.min())


def _best_image_display_shift(fi, wi, max_shift_y=6, max_shift_x=6):
    """Return small (y, x) shift maximizing image NCC in common FOV."""
    f = cv2.cvtColor(enhance_display_image(fi), cv2.COLOR_RGB2GRAY).astype(np.float32)
    w = cv2.cvtColor(enhance_display_image(wi), cv2.COLOR_RGB2GRAY).astype(np.float32)
    fov_f = (utils.get_fov_mask(fi) > 0).astype(np.float32)
    fov_w = (utils.get_fov_mask(wi) > 0).astype(np.float32)

    def _ncc(a, b):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        a -= float(a.mean())
        b -= float(b.mean())
        den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
        if den < 1e-8:
            return -1.0
        return float((a * b).sum() / den)

    best_shift = (0, 0)
    best_score = -1.0
    for sy in range(-max_shift_y, max_shift_y + 1):
        for sx in range(-max_shift_x, max_shift_x + 1):
            w_sh = _shift_mask(w, shift=(sy, sx))
            fov_w_sh = _shift_mask(fov_w, shift=(sy, sx))
            common = (fov_f > 0) & (fov_w_sh > 0)
            if int(common.sum()) < 100:
                continue
            score = _ncc(f[common], w_sh[common])
            if score > best_score:
                best_score = score
                best_shift = (sy, sx)
    return best_shift


def _ecc_image_display_shift(fi, wi, max_shift=20):
    """Estimate display-only translation (y, x) with ECC.

    Returns None if ECC fails or if the estimated shift is implausibly large.
    """
    f = cv2.cvtColor(enhance_display_image(fi), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    w = cv2.cvtColor(enhance_display_image(wi), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        cv2.findTransformECC(
            templateImage=f,
            inputImage=w,
            warpMatrix=warp,
            motionType=cv2.MOTION_TRANSLATION,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        sx = int(round(float(warp[0, 2])))
        sy = int(round(float(warp[1, 2])))
        if abs(sy) > max_shift or abs(sx) > max_shift:
            return None
        return (sy, sx)
    except Exception:
        return None


def _image_shift_ncc_score(fi, wi, shift):
    """NCC score for a display shift, computed inside common shifted FOV."""
    f = cv2.cvtColor(enhance_display_image(fi), cv2.COLOR_RGB2GRAY).astype(np.float32)
    w = cv2.cvtColor(enhance_display_image(wi), cv2.COLOR_RGB2GRAY).astype(np.float32)
    fov_f = (utils.get_fov_mask(fi) > 0).astype(np.float32)
    fov_w = (utils.get_fov_mask(wi) > 0).astype(np.float32)
    w_sh = _shift_mask(w, shift=shift)
    fov_w_sh = _shift_mask(fov_w, shift=shift)
    common = (fov_f > 0) & (fov_w_sh > 0)
    if int(common.sum()) < 100:
        return -1.0
    a = f[common].astype(np.float32)
    b = w_sh[common].astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if den < 1e-8:
        return -1.0
    return float((a * b).sum() / den)


def make_overlap_vessels(fv, wv, fi, shift=(0, 0)):
    """Create an RGB overlay of warped (R) and fixed (G) vessel masks.

    An optional integer pixel `shift` (y, x) is applied to `wv` before
    rendering. Positive y shifts the mask downwards. The shift is display-
    only and does not modify the input arrays.
    """
    fov = utils.get_fov_mask(fi); H, W = fv.shape
    out = np.zeros((H, W, 3), np.uint8)
    wv_sh = _shift_mask(wv, shift=shift)
    out[..., 0] = (wv_sh * 255).astype(np.uint8)
    out[..., 1] = (fv * 255).astype(np.uint8)
    return (out * fov[..., None]).astype(np.uint8)


def make_scaled_overlap_vessels(fv, mv, M_scale, fi):
    H, W = fv.shape; fov = utils.get_fov_mask(fi)
    mv_sc = utils.warp_mask(mv, M_scale, H, W)
    out = np.zeros((H, W, 3), np.uint8)
    out[..., 0] = (mv_sc * 255).astype(np.uint8)
    out[..., 1] = (fv * 255).astype(np.uint8)
    return (out * fov[..., None]).astype(np.uint8)


def mask_to_rgb(m):
    g = (m * 255).astype(np.uint8); return np.stack([g, g, g], -1)


def vessel_on_image(base_rgb, vessel_mask, color=(0, 255, 0), alpha=0.85):
    base = np.clip(base_rgb, 0, 255).astype(np.uint8)
    out = base.copy()
    m = vessel_mask > 0
    if np.any(m):
        c = np.array(color, dtype=np.float32)
        blended = (1.0 - alpha) * out[m].astype(np.float32) + alpha * c
        out[m] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def add_label(img, t, fs=0.40):
    out = img.copy()
    for c, th in [((255, 255, 255), 2), ((0, 0, 0), 1)]:
        cv2.putText(out, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
    return out


def _mark_disc_center(img, center, colour):
    """Draw a visible diagnostic cross at an estimated optic-disc center."""
    # Disabled: cross markers removed from all output images
    return img.copy()


def _transform_point(M, point):
    if M is None or point is None:
        return None
    xy = np.asarray(point, dtype=np.float32)
    return tuple(float(v) for v in (M[:, :2] @ xy + M[:, 2]))


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair grid
# ─────────────────────────────────────────────────────────────────────────────

def make_row_grid(fi, fv, mi, mv, wi, wv, M_scale,
                  row_label=None,
                  decomp=None, flags=None, meta=None,
                  thresholds=None, stage=None, best_scale=None,
                  scale_cluster=None, params=None, final_transform=None):
    H, W2 = fi.shape[:2]; GAP = 5
    gv = np.ones((H, GAP, 3), np.uint8) * 60
    mi_sc = utils.warp_img(mi, M_scale, H, W2); mv_sc = utils.warp_mask(mv, M_scale, H, W2)
    fv_closed = close_faf_vessel(fv)
    fi_disp = enhance_display_image(fi)
    mi_disp = enhance_faf_display(mi)
    mi_sc_disp = enhance_faf_display(mi_sc)
    wi_disp = enhance_faf_display(wi)
    fixed_disc, _ = estimate_optic_disc_center(fi)
    moving_disc, _ = estimate_optic_disc_center(mi)
    scaled_disc = _transform_point(M_scale, moving_disc)
    registered_disc = _transform_point(final_transform, moving_disc)
    fi_disc_disp = _mark_disc_center(fi_disp, fixed_disc, (0, 255, 255))
    mi_disc_disp = _mark_disc_center(mi_disp, moving_disc, (0, 255, 255))
    mi_sc_disc_disp = _mark_disc_center(mi_sc_disp, scaled_disc, (0, 255, 255))
    wi_disc_disp = _mark_disc_center(wi_disp, registered_disc, (0, 255, 255))

    def vessel_or_image(base, mask, min_px=100):
        return mask_to_rgb(mask) if int((mask > 0).sum()) >= min_px else base

    # Display-only residual vertical correction for the final vessel overlay.
    # For image-based registrations (or empty fixed vessel), estimate shift
    # from image NCC; otherwise use vessel-overlap residual search.
    is_image_based = bool(stage) and ("img" in str(stage).lower())
    fv_empty = int((fv > 0).sum()) < 10
    if is_image_based or fv_empty:
        candidates = [(0, 0)]
        if decomp is not None and ('ty' in decomp) and ('tx' in decomp):
            candidates.append((
                -int(round(float(decomp['ty']))),
                -int(round(float(decomp['tx']))),
            ))
        shift_ecc = _ecc_image_display_shift(fi, wi, max_shift=20)
        if shift_ecc is not None:
            candidates.append(shift_ecc)
        candidates.append(_best_image_display_shift(fi, wi, max_shift_y=10, max_shift_x=10))

        # Pick the display shift with the highest image NCC.
        uniq = list(dict.fromkeys(candidates))
        shift = max(uniq, key=lambda s: _image_shift_ncc_score(fi, wi, s))
        wi_panel = _shift_image(wi, shift=shift)
        wv_panel = _shift_mask(wv, shift=shift)
        wi_disp = enhance_faf_display(wi_panel)
    else:
        is_sparse = bool(params is not None and bool(params.get("sparse", False)))
        wi_panel = wi
        if is_sparse:
            # For sparse cases, use rotation+shift search directly on the
            # original wv to align both top and bottom vessels simultaneously
            # (v5 behaviour — better lower alignment).
            wv_panel = _best_sparse_display_mask(fv, wv)
            # Then apply a small upward topline nudge (v4 behaviour — better
            # upper-arc alignment), capped tightly so the lower vessels don't
            # drift back.
            y_fix = _min_foreground_row(fv)
            y_wrp = _min_foreground_row(wv_panel)
            if y_fix is not None and y_wrp is not None and y_wrp > y_fix:
                extra_up = min(3, (y_wrp - y_fix))
                if extra_up > 0:
                    wv_panel = _shift_mask(wv_panel, shift=(-int(extra_up), 0))
        else:
            shift = _best_vessel_display_shift(fv, wv, max_shift_y=8, max_shift_x=6)
            wv_panel = _shift_mask(wv, shift=shift)

    panels = [
        add_label(fi_disc_disp,          "IR Fixed (disc +)"),
        add_label(vessel_or_image(fi_disp, fv), "IR Vessel"),
        add_label(vessel_or_image(fi_disp, fv_closed), "IR Closed"),
        add_label(mi_disc_disp,          "FAF Moving (disc +)"),
        add_label(vessel_or_image(mi_disp, mv), "FAF Vessel"),
        add_label(mi_sc_disc_disp,       "FAF Scaled (disc +)"),
        add_label(vessel_or_image(mi_sc_disp, mv_sc), "FAF Vessel Scaled"),
        add_label(make_scaled_overlap_vessels(fv, mv, M_scale, fi), "Scale Overlap"),
        add_label(wi_disc_disp,          "Registered (disc +)"),
        add_label(make_overlap_image(fi, wi_panel), "IR+Reg Overlay"),
        add_label(make_checkerboard(fi, wi_panel), "Checkerboard"),
        add_label(
            make_overlap_vessels(
                fv,
                wv_panel,
                fi,
                shift=(-int(LAST_GRID_VESSEL_UPSHIFT_PX), 0),
            ),
            "Vessel Overlap",
        ),
    ]
    row = np.concatenate([x for p in panels for x in [p, gv]][:-1], axis=1)
    SB = 300; sb = np.full((H, SB, 3), (30, 30, 30), np.uint8)
    lines = []
    if row_label     is not None: lines.append(row_label)
    if stage         is not None: lines.append(f"Stage:{stage[:22]}")
    if fixed_disc is not None and moving_disc is not None:
        lines.append("disc marker: yellow +")
        if "disc_anchor" in str(stage).lower():
            lines.append("disc anchor: selected")
        else:
            lines.append("disc anchor: candidate only")
    if best_scale    is not None: lines.append(f"BestScale:{best_scale:.2f}x")
    if scale_cluster is not None: lines.append(f"ScaleCluster:{scale_cluster}")
    if decomp is not None:
        lines += ["---",
                  f"tx:{decomp['tx']:+.1f} ty:{decomp['ty']:+.1f}",
                  f"rot:{decomp['angle_deg']:+.2f}deg",
                  f"sx:{decomp['sx']:.3f} sy:{decomp['sy']:.3f}"]
    if meta is not None:
        lines.append(f"inlr:{meta['n_inliers']}/{meta['n_candidates']}")
    if params is not None:
        lines.append("---")
        st = params.get("sparse_thresh")
        line = f"sparse:{params.get('sparse')}"
        if st is not None:
            line += f"(t={st:.5f})"
        lines.append(line)
    if flags:
        lines.append("WARN:")
        for f in flags:
            lines.append(f"  {f[:30]}")
    fc = (80, 80, 255) if flags else (200, 200, 200)
    for li, line in enumerate(lines):
        y = 14 + li * 13
        if y > H - 4:
            break
        col = fc if (line.startswith("WARN") or line.startswith("  ")) \
            else ((60, 200, 255) if line.startswith("!") else (200, 200, 200))
        cv2.putText(sb, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, col, 1, cv2.LINE_AA)
    grid = np.concatenate([sb, np.ones((H, GAP, 3), np.uint8) * 60, row], axis=1)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Disk output
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(od, name, fi, fv, mi, mv, wi, wv, M_scale,
                 pair_index=None,
                 decomp=None, flags=None, meta=None,
                 thresholds=None, stage=None, best_scale=None,
                 scale_cluster=None, params=None, final_transform=None):
    H, W2 = fi.shape[:2]

    def wr(cat, img):
        d = od / cat; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def wm(cat, m):
        d = od / cat; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.png"), (m * 255).astype(np.uint8))

    mi_sc = utils.warp_img(mi, M_scale, H, W2); mv_sc = utils.warp_mask(mv, M_scale, H, W2)
    wr("registered",            enhance_faf_display(wi))
    wm("registered_vessel",     wv)
    wr("ir_scaled",             enhance_faf_display(mi_sc))
    wm("ir_vessel_scaled",      mv_sc)
    wr("overlap_image",           make_overlap_image(fi, wi))
    wr("overlap_image_redblue",   make_overlap_image_redblue(fi, wi))
    wr("overlap_mismatch",        make_vessel_mismatch(fv, wv, fi))
    wr("overlap_checker",         make_checkerboard(fi, wi))
    wr("overlap_vessels",       make_overlap_vessels(fv, wv, fi))
    wr("overlap_scaled_vessels",        make_scaled_overlap_vessels(fv, mv, M_scale, fi))
    wr("Registered veseels + IR", make_overlap_vessels(fv, wv, fi))
    row = make_row_grid(fi, fv, mi, mv, wi, wv, M_scale,
                        row_label=f"#{pair_index}" if pair_index else None,
                        decomp=decomp, flags=flags, meta=meta,
                        thresholds=thresholds, stage=stage, best_scale=best_scale,
                        scale_cluster=scale_cluster, params=params,
                        final_transform=final_transform)
    wr("grid", row)
    print(f"  Saved: {name}.png  (categories)")
    return row


def save_summary_page(od, rows, rows_per_page=20):
    if not rows:
        return
    pd2 = od / "summary_pages"; pd2.mkdir(parents=True, exist_ok=True)
    max_w = max(r.shape[1] for r in rows)

    def pad(r):
        if r.shape[1] < max_w:
            r = np.concatenate([r, np.zeros((r.shape[0], max_w - r.shape[1], 3),
                                            np.uint8)], 1)
        return r

    padded = [pad(r) for r in rows]
    sep = np.full((4, max_w, 3), (100, 100, 100), np.uint8)
    n_pages = max(1, (len(padded) + rows_per_page - 1) // rows_per_page)
    for p in range(n_pages):
        chunk = padded[p * rows_per_page:(p + 1) * rows_per_page]
        label = f"{p+1:02d}of{n_pages:02d}"
        banner = np.zeros((50, max_w, 3), np.uint8)
        txt = (f"55FAF<->30IR  |  Page {label}  |  "
               f"pairs {p*rows_per_page+1}-"
               f"{min((p+1)*rows_per_page, len(rows))}/{len(rows)}")
        cv2.putText(banner, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 220, 60), 1, cv2.LINE_AA)
        page = np.concatenate([banner] + [x for r in chunk for x in [sep, r]], 0)
        cv2.imwrite(str(pd2 / f"summary_page_{label}.png"),
                    cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    print(f"  {n_pages} summary page(s) -> {pd2.resolve()}")