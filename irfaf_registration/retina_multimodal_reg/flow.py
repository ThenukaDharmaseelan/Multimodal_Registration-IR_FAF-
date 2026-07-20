
"""fafir_registration.flow — optical-flow refinement of an affine estimate.

Dense TV-L1 (or Farneback) optical flow is fit to a local affine and composed
onto the running transform. Provides the multi-resolution flow refiner, the
iterative compositional refiner, the anatomical-score acceptance gate, and the
``cascade_refine`` driver that chains them.

Every refinement step is guarded by :func:`fafir_registration.utils.sane` and
only accepted if it does not lower the anatomical score, so refinement can never
degrade a candidate below its pre-refinement quality.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import utils
from .metrics import anatomical_score
from .preprocessing import close_faf_vessel


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance gate
# ─────────────────────────────────────────────────────────────────────────────

def _accept_refinement(fv, mv, M_old, M_new, params=None,
                       eps=None, topo_tol=0.01, bounds=None):
    if eps is None:
        eps = utils.REFINEMENT_EPS
    if not utils.sane(M_new, params, bounds):
        return False, anatomical_score(M_old, fv, mv, params)
    a_old = anatomical_score(M_old, fv, mv, params)
    a_new = anatomical_score(M_new, fv, mv, params)
    return (a_new - a_old >= eps), a_new


# ─────────────────────────────────────────────────────────────────────────────
# Optical flow — multi-resolution
# ─────────────────────────────────────────────────────────────────────────────

def _run_tvl1(src8, dst8):
    try:
        of = cv2.optflow.DualTVL1OpticalFlow_create()
        return of.calc(src8, dst8, None)
    except Exception:
        return cv2.calcOpticalFlowFarneback(src8, dst8, None,
                                            0.5, 3, 25, 5, 7, 1.5, 0)


def flow_to_affine_from_field(fv, flow, M_init, params=None):
    H, W = fv.shape
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    q25 = float(np.percentile(mag, 25))
    q75 = float(np.percentile(mag, 75))
    iqr = q75 - q25
    iqr_bound = float(np.median(mag)) + 1.5 * iqr
    max_disp = float(max(iqr_bound, 1.0))
    flow = cv2.GaussianBlur(flow, (31, 31), 10)
    flow[..., 0] = np.clip(flow[..., 0], -max_disp, max_disp)
    flow[..., 1] = np.clip(flow[..., 1], -max_disp, max_disp)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fv_cl = close_faf_vessel(fv)
    vmask = cv2.dilate((fv_cl > 0).astype(np.uint8), k).astype(np.float32)
    flow[..., 0] *= vmask; flow[..., 1] *= vmask
    step  = 10
    gy, gx = np.mgrid[0:H:step, 0:W:step]
    sp    = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)
    fx    = flow[::step, ::step, 0].ravel()
    fy    = flow[::step, ::step, 1].ravel()
    dp    = sp + np.stack([fx, fy], -1).astype(np.float32)
    valid = np.sqrt(fx ** 2 + fy ** 2) > 0.3
    if valid.sum() < 6:
        return None
    Mf, im = cv2.estimateAffine2D(sp[valid], dp[valid], method=cv2.RANSAC,
                                  ransacReprojThreshold=3.0,
                                  maxIters=2000, confidence=0.999)
    if Mf is None:
        return None
    return utils.compose(M_init, Mf.astype(np.float32))


def tvl1_flow_to_affine_multires(fv, mv, M_init, params=None, bounds=None):
    H, W  = fv.shape
    fv8  = (close_faf_vessel(fv) * 255).astype(np.uint8)
    mv8  = (mv * 255).astype(np.uint8)
    mv_w = cv2.warpAffine(mv8, M_init, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    flow_full = _run_tvl1(mv_w, fv8)
    M_full    = flow_to_affine_from_field(fv, flow_full, M_init, params)
    fv_h  = cv2.resize(fv8,  (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
    mv_wh = cv2.resize(mv_w, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
    flow_half = _run_tvl1(mv_wh, fv_h)
    flow_up   = cv2.resize(flow_half, (W, H), interpolation=cv2.INTER_LINEAR) * 2.0
    M_half    = flow_to_affine_from_field(fv, flow_up, M_init, params)
    M_comp_flow = None
    if M_half is not None and utils.sane(M_half, params, bounds):
        mv_w2       = cv2.warpAffine(mv8, M_half, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        flow2       = _run_tvl1(mv_w2, fv8)
        M_comp_flow = flow_to_affine_from_field(fv, flow2, M_half, params)
    opts = [(M_full, "full"), (M_half, "half"), (M_comp_flow, "multires")]
    best_M2, best_score2, best_lbl2 = M_init, -1.0, "none"
    for M_opt, lbl in opts:
        if M_opt is not None and utils.sane(M_opt, params, bounds):
            sc = anatomical_score(M_opt, fv, mv, params)
            if sc > best_score2:
                best_score2, best_M2, best_lbl2 = sc, M_opt, lbl
    print(f"  [MultiResFlow] best={best_lbl2}  anat={best_score2:.3f}")
    return best_M2, best_score2


def compositional_flow_refine(fv, mv, M_init, n_iters=3, params=None, bounds=None):
    H, W  = fv.shape
    fv8  = (close_faf_vessel(fv) * 255).astype(np.uint8)
    mv8  = (mv * 255).astype(np.uint8)
    M_cur = M_init
    a_cur = anatomical_score(M_cur, fv, mv, params)
    for i in range(n_iters):
        mv_w  = cv2.warpAffine(mv8, M_cur, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        flow  = _run_tvl1(mv_w, fv8)
        M_next = flow_to_affine_from_field(fv, flow, M_cur, params)
        if M_next is None:
            break
        if not utils.sane(M_next, params, bounds):
            break
        accept, a_next = _accept_refinement(fv, mv, M_cur, M_next, params, bounds=bounds)
        print(f"  [CompFlow iter {i+1}] anat {a_cur:.3f} -> {a_next:.3f}  "
              f"(accept={accept})")
        if not accept:
            break
        M_cur, a_cur = M_next, a_next
    return M_cur, anatomical_score(M_cur, fv, mv, params)


# ─────────────────────────────────────────────────────────────────────────────
# Cascade driver
# ─────────────────────────────────────────────────────────────────────────────

def cascade_refine(fv, mv, M_in, label_in, params=None,
                   no_multires_flow=False, no_comp_flow=False, bounds=None):
    d_cur = anatomical_score(M_in, fv, mv, params)
    M_cur, lbl_cur = M_in, label_in
    if not no_multires_flow:
        M_flow, d_flow = tvl1_flow_to_affine_multires(fv, mv, M_cur, params, bounds=bounds)
    else:
        M_flow = None
    if utils.sane(M_flow, params, bounds):
        accept, _ = _accept_refinement(fv, mv, M_cur, M_flow, params, bounds=bounds)
        if accept:
            print(f"  [Cascade] Flow: {d_cur:.3f} -> {d_flow:.3f}")
            M_cur, d_cur, lbl_cur = M_flow, d_flow, lbl_cur + "+flow"
    if not no_comp_flow:
        M_comp, d_comp = compositional_flow_refine(fv, mv, M_cur, n_iters=3,
                                                   params=params, bounds=bounds)
    else:
        M_comp = None
    if utils.sane(M_comp, params, bounds):
        accept, _ = _accept_refinement(fv, mv, M_cur, M_comp, params, bounds=bounds)
        if accept:
            print(f"  [Cascade] CompFlow: {d_cur:.3f} -> {d_comp:.3f}")
            M_cur, d_cur, lbl_cur = M_comp, d_comp, lbl_cur + "+comp"
    return M_cur, d_cur, lbl_cur

