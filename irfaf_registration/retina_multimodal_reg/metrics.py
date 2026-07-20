
"""fafir_registration.metrics — registration-quality measures.

Vessel-mask metrics (Dice, NCC, HD95/ASD surface distance, Wasserstein optimal
transport, centerline recall, connectivity), the composite scoring functions
used to rank candidates (``anatomical_score``, ``asd_reward``), oracle
candidate diagnostics, and image-intensity metrics (MI/NMI/NCC/SSIM/PSNR).

All numerical behaviour is identical to the original validated engine, with one
determinism fix: ``_sliced_wasserstein_fallback`` now resamples with its own
seeded RNG instead of the global ``np.random``, so candidate scores (and thus
the winning transform) are reproducible run-to-run. The fixed configuration
constant ``ASD_DIRECTED`` is read from :mod:`fafir_registration.utils` so a
single assignment stays authoritative.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from . import utils
from .preprocessing import skeletonize, vessel_density

# Optional POT (python-optimal-transport) for exact Wasserstein.
try:
    import ot as _ot
    _POT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ot = None
    _POT_AVAILABLE = False

# Optional scikit-image for SSIM.
try:
    from skimage.metrics import structural_similarity as _sk_ssim
    _SSIM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SSIM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Simple overlap metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice(a, b):
    return float(2 * (a * b).sum() / (a.sum() + b.sum() + 1e-8))


def ncc(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean();                 b = b - b.mean()
    da = float(np.sqrt((a * a).sum())); db = float(np.sqrt((b * b).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


# ─────────────────────────────────────────────────────────────────────────────
# Whole-structure vessel distances
# ─────────────────────────────────────────────────────────────────────────────

def hausdorff_vessel_distance(fv, wv, max_pts=3000, seed=0):
    H, W     = fv.shape
    img_diag = float(np.sqrt(H ** 2 + W ** 2))
    rng      = np.random.default_rng(seed)

    def _skel_pts(v, n):
        pts = np.argwhere(skeletonize(v) > 0).astype(np.float32)
        if len(pts) > n:
            idx = rng.choice(len(pts), n, replace=False)
            pts = pts[idx]
        return pts

    pts_f = _skel_pts(fv, max_pts)
    pts_w = _skel_pts(wv, max_pts)
    if len(pts_f) < 3 or len(pts_w) < 3:
        return dict(hd95=None, hd95_fwd=None, hd95_bwd=None,
                    hd95_norm=None, masd=None, masd_norm=None,
                    asd_fwd=None, asd_bwd=None,
                    asd_fwd_norm=None, asd_bwd_norm=None)
    D      = cdist(pts_f, pts_w)
    nn_fw  = D.min(axis=1)
    nn_bw  = D.min(axis=0)
    pooled = np.concatenate([nn_fw, nn_bw])
    hd95     = float(np.percentile(pooled, 95))
    hd95_fwd = float(np.percentile(nn_fw,  95))
    hd95_bwd = float(np.percentile(nn_bw,  95))
    masd     = float(pooled.mean())     # symmetric (pooled) surface distance == ASD
    asd_fwd  = float(nn_fw.mean())      # directed ASD: fixed -> warped
    asd_bwd  = float(nn_bw.mean())      # directed ASD: warped -> fixed
    return dict(
        hd95       = round(hd95,     2),
        hd95_fwd   = round(hd95_fwd, 2),
        hd95_bwd   = round(hd95_bwd, 2),
        hd95_norm  = round(hd95 / img_diag,  4),
        masd       = round(masd,     2),
        masd_norm  = round(masd / img_diag,  4),
        asd_fwd      = round(asd_fwd, 2),
        asd_bwd      = round(asd_bwd, 2),
        asd_fwd_norm = round(asd_fwd / img_diag, 4),
        asd_bwd_norm = round(asd_bwd / img_diag, 4),
    )


def _sliced_wasserstein_fallback(pts_f, pts_w, n_proj=200, seed=42):
    rng  = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, 2)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sw = 0.0
    for d in dirs:
        p1 = pts_f @ d
        p2 = pts_w @ d
        # Resample with the SEEDED rng (not global np.random) so the result is
        # deterministic — this keeps candidate scores reproducible run-to-run.
        p2r = rng.choice(p2, len(p1), replace=len(p2) < len(p1))
        sw += float(np.mean(np.abs(np.sort(p1) - np.sort(p2r))))
    return sw / n_proj


def wasserstein_vessel_distance(fv, wv, max_pts=2000, n_proj=200,
                                use_skeleton=True):
    H, W     = fv.shape
    img_diag = float(np.sqrt(H ** 2 + W ** 2))

    def _skel_pts(v, n):
        src = (skeletonize(v) > 0) if use_skeleton else (v > 0)
        pts = np.argwhere(src).astype(np.float32)
        if len(pts) > n:
            idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
            pts = pts[idx]
        return pts

    pts_f = _skel_pts(fv, max_pts)
    pts_w = _skel_pts(wv, max_pts)
    null = dict(wass_w1=None, wass_w2=None,
                wass_w1_norm=None, wass_w2_norm=None,
                wass_sliced_w1=None, wass_pot_used=False)
    if len(pts_f) < 3 or len(pts_w) < 3:
        return null
    sliced = _sliced_wasserstein_fallback(pts_f, pts_w, n_proj=n_proj)
    if _POT_AVAILABLE:
        try:
            nf, nw  = len(pts_f), len(pts_w)
            a = np.ones(nf, dtype=np.float64) / nf
            b = np.ones(nw, dtype=np.float64) / nw
            M1  = cdist(pts_f, pts_w, metric="cityblock").astype(np.float64)
            w1  = float(_ot.emd2(a, b, M1))
            M2  = cdist(pts_f, pts_w, metric="sqeuclidean").astype(np.float64)
            w2  = float(np.sqrt(max(_ot.emd2(a, b, M2), 0.0)))
            return dict(
                wass_w1        = round(w1,     2),
                wass_w2        = round(w2,     2),
                wass_w1_norm   = round(w1 / img_diag, 4),
                wass_w2_norm   = round(w2 / img_diag, 4),
                wass_sliced_w1 = round(sliced, 2),
                wass_pot_used  = True,
            )
        except Exception as e:  # pragma: no cover
            print(f"  [Wasserstein] POT failed ({e}) — using sliced fallback")
    return dict(
        wass_w1        = round(sliced, 2),
        wass_w2        = None,
        wass_w1_norm   = round(sliced / img_diag, 4),
        wass_w2_norm   = None,
        wass_sliced_w1 = round(sliced, 2),
        wass_pot_used  = False,
    )


def ssd_vessel_distance(fv, X):
    H, W = fv.shape
    diff = fv.astype(np.float32) - X.astype(np.float32)
    ssd  = float(np.sum(diff ** 2))
    return dict(ssd=round(ssd, 2), ssd_mse=round(ssd / float(H * W), 6))


def centerline_overlap_score(fv, wv, tolerances=(2, 4, 8)):
    skel_f = (skeletonize(fv) > 0).astype(np.uint8)
    skel_w = (skeletonize(wv) > 0).astype(np.uint8)
    result  = {}
    recalls = []
    for t in tolerances:
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
        dilated = cv2.dilate(skel_w, k)
        n_f     = float(skel_f.sum())
        if n_f < 1:
            result[f"cl_overlap_t{t}"] = None
            continue
        covered = float((skel_f & dilated).sum())
        recall  = covered / n_f
        result[f"cl_overlap_t{t}"] = round(recall, 4)
        recalls.append(recall)
    result["cl_recall_mean"] = round(float(np.mean(recalls)), 4) if recalls else None
    return result


def vessel_connectivity_score(fv, wv):
    def _cc(v):
        b = (v * 255).astype(np.uint8)
        n, _, _, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        return n - 1

    def _euler(v):
        b   = (v > 0).astype(np.uint8)
        q1  = int(((b[:-1, :-1] == 1) & (b[:-1, 1:] == 0) & (b[1:, :-1] == 0) & (b[1:, 1:] == 0)).sum())
        q1 += int(((b[:-1, :-1] == 0) & (b[:-1, 1:] == 1) & (b[1:, :-1] == 0) & (b[1:, 1:] == 0)).sum())
        q1 += int(((b[:-1, :-1] == 0) & (b[:-1, 1:] == 0) & (b[1:, :-1] == 1) & (b[1:, 1:] == 0)).sum())
        q1 += int(((b[:-1, :-1] == 0) & (b[:-1, 1:] == 0) & (b[1:, :-1] == 0) & (b[1:, 1:] == 1)).sum())
        q3  = int(((b[:-1, :-1] == 0) & (b[:-1, 1:] == 1) & (b[1:, :-1] == 1) & (b[1:, 1:] == 1)).sum())
        q3 += int(((b[:-1, :-1] == 1) & (b[:-1, 1:] == 0) & (b[1:, :-1] == 1) & (b[1:, 1:] == 1)).sum())
        q3 += int(((b[:-1, :-1] == 1) & (b[:-1, 1:] == 1) & (b[1:, :-1] == 0) & (b[1:, 1:] == 1)).sum())
        q3 += int(((b[:-1, :-1] == 1) & (b[:-1, 1:] == 1) & (b[1:, :-1] == 1) & (b[1:, 1:] == 0)).sum())
        qd  = int(((b[:-1, :-1] == 1) & (b[:-1, 1:] == 0) & (b[1:, :-1] == 0) & (b[1:, 1:] == 1)).sum())
        qd += int(((b[:-1, :-1] == 0) & (b[:-1, 1:] == 1) & (b[1:, :-1] == 1) & (b[1:, 1:] == 0)).sum())
        return (q1 - q3 + 2 * qd) // 4

    cc_f  = _cc(fv)
    cc_w  = _cc(wv)
    denom = max(cc_f, cc_w, 1)
    cc_score = 1.0 - abs(cc_w - cc_f) / denom
    eu_f = _euler(fv)
    eu_w = _euler(wv)
    return dict(
        conn_cc_fixed      = cc_f,
        conn_cc_aligned    = cc_w,
        conn_cc_score      = round(float(cc_score), 4),
        conn_euler_fixed   = int(eu_f),
        conn_euler_aligned = int(eu_w),
        conn_euler_delta   = int(eu_w - eu_f),
    )


def compute_all_topology_metrics(M, fv, mv, params=None):
    H, W = fv.shape
    wv   = utils.warp_mask(mv, M, H, W)
    out  = {}
    out.update(hausdorff_vessel_distance(fv, wv))
    out.update(wasserstein_vessel_distance(fv, wv))
    out.update(centerline_overlap_score(fv, wv, tolerances=(2, 4, 8)))
    out.update(vessel_connectivity_score(fv, wv))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Composite scoring / selection objectives
# ─────────────────────────────────────────────────────────────────────────────

def _wass_reward(fv, wv):
    wb = wasserstein_vessel_distance(fv, wv)
    wn = wb.get("wass_w2_norm")
    if wn is None:
        wn = wb.get("wass_w1_norm")
    if wn is None:
        return None
    return float(np.exp(-float(wn)))


def anatomical_score(M, fv, mv, params=None):
    wv = utils.warp_mask(mv, M, *fv.shape)
    conn = vessel_connectivity_score(fv, wv)["conn_cc_score"]
    cl   = centerline_overlap_score(fv, wv).get("cl_recall_mean") or 0.0
    masd_norm = hausdorff_vessel_distance(fv, wv).get("masd_norm")

    asd_r  = (1.0 - min(float(masd_norm), 1.0)) if masd_norm is not None else None
    wass_r = _wass_reward(fv, wv)

    terms = [(0.40, cl), (0.10, conn)]
    if asd_r  is not None: terms.append((0.25, asd_r))
    if wass_r is not None: terms.append((0.25, wass_r))
    w_sum = sum(w for w, _ in terms)
    return float(sum(w * v for w, v in terms) / (w_sum + 1e-8))


def asd_reward(M, fv, mv, params=None):
    H, W = fv.shape
    wv = utils.warp_mask(mv, M, H, W)
    hd = hausdorff_vessel_distance(fv, wv)
    mn = hd.get("asd_fwd_norm") if utils.ASD_DIRECTED else hd.get("masd_norm")
    if mn is None:
        return None
    return 1.0 - min(float(mn), 1.0)


def oracle_candidate_metrics(pool, chosen_lbl, fv, mv):
    rows = []
    for lbl, M in pool:
        wv = utils.warp_mask(mv, M, *fv.shape)
        dsc = float(dice(fv, wv))
        asd = hausdorff_vessel_distance(fv, wv).get("masd")
        wb  = wasserstein_vessel_distance(fv, wv)
        wss = wb.get("wass_w2") if wb.get("wass_w2") is not None else wb.get("wass_w1")
        rows.append((lbl, dsc, asd, wss))
    if not rows:
        return None

    def _argbest(key, lower):
        vals = [(l, k) for l, *m in rows
                for k in [m[key]] if k is not None]
        if not vals:
            return (None, None)
        return min(vals, key=lambda x: x[1]) if lower \
            else max(vals, key=lambda x: x[1])

    best_dice = _argbest(0, lower=False)   # higher Dice is better
    best_asd  = _argbest(1, lower=True)
    best_wass = _argbest(2, lower=True)
    chosen = next((r for r in rows if r[0] == chosen_lbl), None)
    out = dict(
        oracle_best_dice_lbl=best_dice[0], oracle_best_dice=best_dice[1],
        oracle_best_asd_lbl=best_asd[0],   oracle_best_asd=best_asd[1],
        oracle_best_wass_lbl=best_wass[0], oracle_best_wass=best_wass[1],
        chosen_lbl=chosen_lbl,
        chosen_dice=chosen[1] if chosen else None,
        chosen_asd=chosen[2]  if chosen else None,
        chosen_wass=chosen[3] if chosen else None,
    )
    if chosen and best_dice[1] is not None and chosen[1] is not None:
        out["oracle_regret_dice"] = round(best_dice[1] - chosen[1], 4)
    if chosen and best_asd[1] is not None and chosen[2] is not None:
        out["oracle_regret_asd"]  = round(chosen[2] - best_asd[1], 4)
    if chosen and best_wass[1] is not None and chosen[3] is not None:
        out["oracle_regret_wass"] = round(chosen[3] - best_wass[1], 4)
    out["oracle_chosen_is_best_dice"] = bool(chosen_lbl == best_dice[0])
    out["oracle_chosen_is_best_asd"]  = bool(chosen_lbl == best_asd[0])
    out["oracle_chosen_is_best_wass"] = bool(chosen_lbl == best_wass[0])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Common-FOV vessel metrics
# ─────────────────────────────────────────────────────────────────────────────

def common_fov_metrics(fi, wi, fv, wv):
    fov_f = utils.get_fov_mask(fi)
    fov_w = utils.get_fov_mask(wi)
    common = ((fov_f > 0) & (fov_w > 0)).astype(np.float32)
    fv_c = fv * common
    wv_c = wv * common
    hd = hausdorff_vessel_distance(fv_c, wv_c)
    wb = wasserstein_vessel_distance(fv_c, wv_c)
    return dict(
        common_fov_frac = round(float(common.mean()), 4),
        dsc_after_cfov  = round(dice(fv_c, wv_c), 4),
        hd95_after_cfov = hd.get("hd95"),
        asd_after_cfov  = hd.get("masd"),
        wass_after_cfov = wb.get("wass_w2") if wb.get("wass_w2") is not None else wb.get("wass_w1"),
        cl_recall_cfov  = centerline_overlap_score(fv_c, wv_c).get("cl_recall_mean"),
        conn_cfov       = vessel_connectivity_score(fv_c, wv_c).get("conn_cc_score"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image-intensity metrics (computed on greyscale images, not masks)
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(img):
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    return g.astype(np.float32) / 255.0


def _mutual_information(a, b, bins=64):
    if a.size < 16:
        return None, None
    hist, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    pab = hist / (hist.sum() + 1e-12)
    pa  = pab.sum(axis=1)
    pb  = pab.sum(axis=0)
    nz  = pab > 0
    h_ab = -np.sum(pab[nz] * np.log2(pab[nz]))
    h_a  = -np.sum(pa[pa > 0]  * np.log2(pa[pa > 0]))
    h_b  = -np.sum(pb[pb > 0]  * np.log2(pb[pb > 0]))
    mi   = h_a + h_b - h_ab
    nmi  = (h_a + h_b) / (h_ab + 1e-12)
    return float(mi), float(nmi)


def image_intensity_metrics(fixed_rgb, moving_rgb, fov_mask=None, bins=64):
    fg = _to_gray(fixed_rgb)
    mg = _to_gray(moving_rgb)

    if fov_mask is not None:
        m = fov_mask > 0
    else:
        m = np.ones(fg.shape, dtype=bool)
    if m.sum() < 16:
        return dict(mi=None, nmi=None, ncc_img=None, ssim=None,
                    mse=None, rmse=None, psnr=None, intensity_fov_frac=0.0)

    fa = fg[m].ravel()
    ma = mg[m].ravel()

    mi, nmi = _mutual_information(fa, ma, bins=bins)

    fz = fa - fa.mean(); mz = ma - ma.mean()
    df = float(np.sqrt((fz * fz).sum())); dm = float(np.sqrt((mz * mz).sum()))
    ncc_img = float((fz * mz).sum() / (df * dm)) if df > 1e-12 and dm > 1e-12 else 0.0

    mse  = float(np.mean((fa - ma) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 1e-12 else None  # data range = 1.0

    if _SSIM_AVAILABLE:
        ssim = float(_sk_ssim(fg, mg, data_range=1.0))
    else:
        ssim = None

    return dict(
        mi                 = round(mi,   4) if mi   is not None else None,
        nmi                = round(nmi,  4) if nmi  is not None else None,
        ncc_img            = round(ncc_img, 4),
        ssim               = round(ssim, 4) if ssim is not None else None,
        mse                = round(mse,  6),
        rmse               = round(rmse, 6),
        psnr               = round(psnr, 3) if psnr is not None else None,
        intensity_fov_frac = round(float(m.mean()), 4),
    )
