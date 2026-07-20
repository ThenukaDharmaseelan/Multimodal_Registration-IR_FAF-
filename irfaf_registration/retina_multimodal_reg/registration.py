"""irfaf_registration.registration — candidate generation and selection.

The heart of the pipeline: seed the transform from the FOV scale, spawn LoFTR-
and flow-refined candidates, gate each by affine sanity, and pick the winner by
the directed-ASD final selector (falling back to the anatomical score). Handles
both the normal and the sparse-vessel regimes, then hands the winner to the
optical-flow cascade for a final polish.

Configuration is read live from :mod:`irfaf_registration.utils` (FOV scale,
scale-search window, multiscale grid, adaptive sanity). The per-pair sanity
bounds computed here are published to ``utils.CURRENT_BOUNDS`` so the shared
:func:`~irfaf_registration.utils.sane` gate sees them without threading an
argument through every call.
"""

from __future__ import annotations

import numpy as np
import cv2

from . import utils
from .flow import cascade_refine, tvl1_flow_to_affine_multires
from .matching import loftr_to_affine, loftr_to_affine_image
from .metrics import anatomical_score, asd_reward, image_intensity_metrics
from .models import loftr
from .preprocessing import (
    estimate_optic_disc_center,
    heuristic_params,
    vessel_density,
)


# ─────────────────────────────────────────────────────────────────────────────
# FOV-scale seed candidate
# ─────────────────────────────────────────────────────────────────────────────

def fov_scale_candidate(fv, mv, params=None):
    H, W = fv.shape
    seed = float(utils.FOV_SCALE_RATIO)
    cx, cy = W / 2.0, H / 2.0

    win_lo, win_hi = utils.scale_search_window(seed)

    if not utils.SCALE_SEARCH_ENABLED:
        s = seed
        M = utils.make_scale_matrix(s, cx, cy)
        d = anatomical_score(M, fv, mv, params)
        print(f"  [FOVScale] fixed FOV seed {s:.3f}x  (anat={d:.3f})  "
              f"— no search")
        sweep = [(round(s, 3), round(float(d), 4))]
        moved = False
    else:
        lo, hi = win_lo, win_hi
        scales = np.round(np.arange(lo, hi + 1e-9, utils.SCALE_SEARCH_STEP), 4)
        scales = np.unique(np.append(scales, round(seed, 4)))

        seed_M = utils.make_scale_matrix(seed, cx, cy)
        seed_d = anatomical_score(seed_M, fv, mv, params)
        best_s, best_M, best_d = seed, seed_M, seed_d

        sweep = []
        for sc in scales:
            M_sc = utils.make_scale_matrix(float(sc), cx, cy)
            sc_score = anatomical_score(M_sc, fv, mv, params)
            sweep.append((float(sc), float(sc_score)))
            if sc_score > best_d:
                best_s, best_M, best_d = float(sc), M_sc, float(sc_score)

        moved = (best_s != seed)
        if moved and (best_d - seed_d) < utils.SCALE_SEARCH_MARGIN:
            best_s, best_M, best_d = seed, seed_M, seed_d
            moved = False

        s, M, d = best_s, best_M, best_d
        top = "  ".join(f"{sc:.2f}:{dd:.3f}"
                        for sc, dd in sorted(sweep, key=lambda x: x[1],
                                             reverse=True)[:5])
        print(f"  [FOVScale] search [{lo:.3f},{hi:.3f}] step "
              f"{utils.SCALE_SEARCH_STEP:.3f}  seed={seed:.3f}(d={seed_d:.3f})  →  "
              f"best={s:.3f}(d={d:.3f})  {'MOVED' if moved else 'kept seed'}  "
              f"top5> {top}")
        sweep = [(round(sc, 3), round(dd, 4)) for sc, dd in sweep]

    scale_meta = dict(
        scale_cluster="fov_search" if utils.SCALE_SEARCH_ENABLED else "fov",
        scale_bimodal=False,
        scale_top3=[(round(s, 2), round(float(d), 4))],
        alt_cluster_M=None,
        alt_cluster_s=None,
        alt_cluster_d=-1.0,
        scale_bifurc_est=round(s, 4),
        scale_geom_est=round(s, 4),
        scale_moved_from_seed=bool(moved),
        scale_window=(round(win_lo, 3), round(win_hi, 3))
                     if utils.SCALE_SEARCH_ENABLED else (round(s, 3), round(s, 3)),
        s_lo=win_lo if utils.SCALE_SEARCH_ENABLED else s,
        s_hi=win_hi if utils.SCALE_SEARCH_ENABLED else s,
        n_window_candidates=len(sweep),
        scale_n_signals=0,
    )
    return M, d, s, scale_meta


# ─────────────────────────────────────────────────────────────────────────────
# LoFTR candidate spawning
# ─────────────────────────────────────────────────────────────────────────────

# Vessel density below this triggers image-based LoFTR as additional candidates.


def _loftr_from_scale(fv, mv, M_sc, scale_lbl, H, W, params=None,
                      no_multires_flow=False, bounds=None,
                      fi=None, mi=None, any_mask_missing=False):
    if not loftr.LOFTR_AVAILABLE:
        return []
    mv_sc = utils.warp_mask(mv, M_sc, H, W)
    results = []

    # Quick mask-quality heuristic: if the warped FAF mask is tiny,
    # extremely fragmented, or dominated by tiny components, treat it as
    # effectively missing so we fall back to image-based LoFTR instead of
    # producing spurious vessel matches.
    try:
        mv_bin = (mv_sc > 0).astype('uint8')
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mv_bin, connectivity=8)
        if n_labels <= 1:
            largest_cc_ratio = 0.0
            total_cc_area = 0.0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            total_cc_area = float(areas.sum()) if areas.size else 0.0
            largest_cc_ratio = float(areas.max()) / max(1.0, total_cc_area) if areas.size else 0.0
    except Exception:
        largest_cc_ratio = 0.0
        total_cc_area = 0.0

    results = []

    # ── vessel-based LoFTR ────────────────────────────────────────────────
    # Skip if either mask is empty or was missing from disk — LoFTR on zero
    # images produces spurious identity matches that block image-based fallback.
    fv_empty = float(fv.sum()) < 1.0
    mv_sc_empty = float(mv_sc.sum()) < 1.0
    # force image-based when mask is missing/empty OR when warped mask is
    # very small or heavily fragmented (low largest-CC ratio)
    force_image_based = (
        any_mask_missing
        or fv_empty
        or mv_sc_empty
        or float(mv_sc.sum()) < 50
        or largest_cc_ratio < 0.05
    )
    if force_image_based:
        print(f"  [LoFTR@{scale_lbl}] skipped (empty/low-quality vessel mask)")
        M_loftr = None
    else:
        M_loftr, n_loftr = loftr_to_affine(fv, mv_sc, params=params, bounds=bounds)
    if M_loftr is None:
        if not force_image_based:
            print(f"  [LoFTR@{scale_lbl}] no matches")
    else:
        M_full = utils.compose(M_sc, M_loftr)
        if not utils.sane(M_full, params, bounds):
            print(f"  [LoFTR@{scale_lbl}] insane affine — skipped")
        else:
            d = anatomical_score(M_full, fv, mv, params)
            print(f"  [LoFTR@{scale_lbl}] n={n_loftr}  d={d:.3f}")
            results.append((f"LoFTR@{scale_lbl}", M_full, d))
            if not no_multires_flow:
                M_flow, d_flow = tvl1_flow_to_affine_multires(
                    fv, mv, M_full, params, bounds=bounds)
                a_base = anatomical_score(M_full, fv, mv, params)
                a_flow = (anatomical_score(M_flow, fv, mv, params)
                          if M_flow is not None else -1.0)
                if utils.sane(M_flow, params, bounds) and a_flow > a_base:
                    print(f"  [LoFTR@{scale_lbl}+flow] anat {a_base:.3f} -> {a_flow:.3f}")
                    results.append((f"LoFTR@{scale_lbl}+flow", M_flow, d_flow))

    # ── image-based LoFTR — fallback for absent *or weak* vessel matching ─
    vessel_score = max((float(d) for _, _, d in results), default=-1.0)
    vessel_match_weak = (not results or
                         vessel_score < utils.IMAGE_FALLBACK_MIN_ANATOMICAL)
    if vessel_match_weak and fi is not None and mi is not None:
        if results:
            print(f"  [LoFTR-img@{scale_lbl}] vessel candidate weak "
                  f"(anat={vessel_score:.3f} < "
                  f"{utils.IMAGE_FALLBACK_MIN_ANATOMICAL:.2f}); trying image fallback")
        mi_sc = utils.warp_img(mi, M_sc, H, W)
        # Missing-mask cases are prone to spurious correspondences; require
        # stronger confidence and more matches for stable affine estimates.
        img_min_conf = 0.35 if any_mask_missing else 0.4
        img_min_matches = 10 if any_mask_missing else 8
        M_img, n_img = loftr_to_affine_image(
            fi,
            mi_sc,
            min_conf=img_min_conf,
            min_matches=img_min_matches,
            params=params,
            bounds=bounds,
        )
        if M_img is None and any_mask_missing:
            M_img, n_img = loftr_to_affine_image(
                fi,
                mi_sc,
                min_conf=0.30,
                min_matches=8,
                params=params,
                bounds=bounds,
            )
        if M_img is not None:
            M_img_full = utils.compose(M_sc, M_img)
            if utils.sane(M_img_full, params, bounds):
                d_img = anatomical_score(M_img_full, fv, mv, params)
                # When vessel masks are empty, anatomical_score returns 0 for
                # every candidate — use a match-count proxy so this candidate
                # beats scale_best and gets selected.
                if d_img == 0.0 and n_img > 0:
                    d_img = min(0.1 + n_img * 0.01, 0.5)
                    print(f"  [LoFTR-img@{scale_lbl}] n={n_img}  d=proxy({d_img:.3f})")
                else:
                    print(f"  [LoFTR-img@{scale_lbl}] n={n_img}  d={d_img:.3f}")
                results.append((f"LoFTR-img@{scale_lbl}", M_img_full, d_img))
                if not no_multires_flow:
                    M_iflow, d_iflow = tvl1_flow_to_affine_multires(
                        fv, mv, M_img_full, params, bounds=bounds)
                    a_ibase = anatomical_score(M_img_full, fv, mv, params)
                    a_iflow = (anatomical_score(M_iflow, fv, mv, params)
                               if M_iflow is not None else -1.0)
                    if utils.sane(M_iflow, params, bounds) and a_iflow > a_ibase:
                        print(f"  [LoFTR-img@{scale_lbl}+flow] anat {a_ibase:.3f} -> {a_iflow:.3f}")
                        results.append(
                            (f"LoFTR-img@{scale_lbl}+flow", M_iflow, d_iflow))

    if not results:
        return []
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Final selection
# ─────────────────────────────────────────────────────────────────────────────

def _final_select(candidates, fv, mv, params=None):
    scored = []
    for lbl, M, d_skel, t_score in candidates:
        r = asd_reward(M, fv, mv, params)
        scored.append((lbl, M, d_skel, t_score, r))

    # If no candidate has a usable ASD reward, fall back to anatomical score.
    if all(r is None for *_, r in scored):
        best = max(scored, key=lambda c: c[3])
        return best[:4], "anatomical(fallback)", None

    scored = [c for c in scored if c[4] is not None]
    scored.sort(key=lambda c: c[4], reverse=True)
    best = scored[0]
    best_asd = best[4]
    close = [c for c in scored if best_asd - c[4] <= utils.ASD_SELECTION_MARGIN]

    if len(close) > 1:
        best_anat = max(close, key=lambda c: c[3])
        if best_anat[0] != best[0]:
            return best_anat[:4], "asd+anatomical", best_anat[4]

    return best[:4], "asd", best_asd


def _image_alignment_score(fi, mi, M, H, W):
    wi = utils.warp_img(mi, M, H, W)
    fov_f = utils.get_fov_mask(fi)
    fov_w = utils.get_fov_mask(wi)
    common = ((fov_f > 0) & (fov_w > 0)).astype(np.uint8)
    m = image_intensity_metrics(fi, wi, fov_mask=common)
    ncc = float(m.get("ncc_img") or 0.0)
    nmi = float(m.get("nmi") or 0.0)
    # NMI is robust to the modality-dependent intensity mapping between IR
    # and FAF. NCC remains only a small tie-breaker for similar NMI values.
    return nmi + 0.05 * ncc


def _disc_anchor_candidate(fi, mi, M_scale, H, W):
    """Return a scale-plus-translation seed that aligns detected disc centers.

    No optic-disc segmentation is required. The landmark is estimated from
    compact bright/dark image extrema and is used only as an additional
    candidate; normal vessel matching remains available if either estimate is
    unreliable or anatomically inconsistent.
    """
    if fi is None or mi is None:
        return None, None
    fixed_center, fixed_conf = estimate_optic_disc_center(fi)
    moving_center, moving_conf = estimate_optic_disc_center(mi)
    if fixed_center is None or moving_center is None:
        return None, None

    moving_xy = np.asarray(moving_center, dtype=np.float32)
    scaled_xy = M_scale[:, :2] @ moving_xy + M_scale[:, 2]
    shift = np.asarray(fixed_center, dtype=np.float32) - scaled_xy
    # A disc candidate beyond this range is likely a false blob, not a useful
    # registration landmark. The broad range still permits eccentric discs.
    if float(np.linalg.norm(shift)) > 0.45 * float(np.hypot(H, W)):
        return None, None

    T = np.array([[1.0, 0.0, shift[0]],
                  [0.0, 1.0, shift[1]]], dtype=np.float32)
    confidence = min(float(fixed_conf), float(moving_conf))
    return utils.compose(M_scale, T), dict(
        fixed_center=fixed_center,
        moving_center=moving_center,
        shift=tuple(float(v) for v in shift),
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-candidate registration driver
# ─────────────────────────────────────────────────────────────────────────────

def register_multi_candidate(fv, mv, H, W,
                             no_loftr=False, no_multires_flow=False,
                             no_comp_flow=False, no_asd_selection=False,
                             no_cascade=False,
                             fi=None, mi=None,
                             any_mask_missing=False):
    fv_density = vessel_density(fv)
    mv_density = vessel_density(mv)
    print(f"  [Density] IR={fv_density:.4f}  FAF={mv_density:.4f}")

    ap = heuristic_params(fv, mv)
    M_scale_best, d_scale_best, best_s, scale_meta = fov_scale_candidate(
        fv, mv, params=ap)

    utils.CURRENT_BOUNDS = utils.seed_sanity_bounds(M_scale_best, fv=fv, mv=mv)
    bounds = utils.CURRENT_BOUNDS
    if utils.ADAPTIVE_SANITY:
        print(f"  [AdaptiveSanity] seed coverage={bounds['seed_coverage']}  "
              f"coverage_floor={bounds['coverage_floor']:.3f}  "
              f"(rotation/scale UNCONSTRAINED; off-frame vessels rejected)")

    print(f"  [Heuristic] sparse={ap['sparse']}  "
          f"density_ratio={ap['density_ratio']:.3f}  "
          f"topo=({ap['topo_weights']['w_cd']:.2f}/"
          f"{ap['topo_weights']['w_sd']:.2f}/"
          f"{ap['topo_weights']['w_bf']:.2f})")

    if ap["sparse"]:
        sparse_side = "IR" if fv_density < ap["sparse_thresh"] else "FAF"
        print(f"  [SPARSE] {sparse_side} density too low "
              f"({fv_density:.4f}/{mv_density:.4f}  "
              f"thresh={ap['sparse_thresh']:.5f})")
        sp = []

        def add_sp(lbl, M, d=None):
            if not utils.sane(M, ap, bounds): return
            d = d if d is not None else anatomical_score(M, fv, mv, ap)
            sp.append((lbl, M, d))

        # Image-only registrations cannot score vessel overlap. Add the
        # no-mask disc landmark transform explicitly so the subsequent NMI
        # selector can compare it against image-LoFTR candidates.
        M_disc, disc_meta = _disc_anchor_candidate(fi, mi, M_scale_best, H, W)
        if M_disc is not None and utils.sane(M_disc, ap, bounds):
            add_sp("disc_anchor(sp)", M_disc)
            print("  [DiscAnchor] "
                  f"fixed=({disc_meta['fixed_center'][0]:.1f},{disc_meta['fixed_center'][1]:.1f}) "
                  f"moving=({disc_meta['moving_center'][0]:.1f},{disc_meta['moving_center'][1]:.1f}) "
                  f"shift=({disc_meta['shift'][0]:+.1f},{disc_meta['shift'][1]:+.1f})")

        # For missing-mask sparse cases, prioritize feature-matching candidates
        # first; keep scale_best only as a hard fallback when none are found.
        if not any_mask_missing:
            add_sp("scale_best(sp)", M_scale_best, d_scale_best)
        if not no_loftr:
            sparse_starts = []
            if any_mask_missing:
                cx, cy = W / 2.0, H / 2.0
                seed_scale = float(utils.FOV_SCALE_RATIO)
                lo, hi = utils.scale_search_window(seed_scale)
                # Fixed compact sweep so missing-mask sparse cases do not get
                # trapped at a single seed scale.
                sweep = [float(lo), float((lo + hi) * 0.5), float(hi), float(seed_scale), float(best_s)]
                for s in sorted(set(round(x, 3) for x in sweep)):
                    M_s = utils.make_scale_matrix(float(s), cx, cy)
                    sparse_starts.append((M_s, f"{s:.2f}(sp)"))
                print(f"  [SparseMultiScale] starts {[lbl for _, lbl in sparse_starts]}")
            else:
                sparse_starts.append((M_scale_best, f"{best_s:.2f}(sp)"))

            for M_start, start_lbl in sparse_starts:
                for lbl, M, d in _loftr_from_scale(fv, mv, M_start,
                                                   start_lbl, H, W, ap,
                                                   no_multires_flow=no_multires_flow,
                                                   bounds=bounds,
                                                   fi=fi, mi=mi,
                                                   any_mask_missing=any_mask_missing):
                    add_sp(lbl, M, d)
        if not sp:
            print(f"  [Sparse] WARNING — all candidates failed sanity; "
                  f"force-adding scale_best as fallback")
            d_forced = anatomical_score(M_scale_best, fv, mv, ap)
            sp.append(("scale_best(sp,forced)", M_scale_best, d_forced))
        # In missing-mask sparse mode, vessel-based scores are uninformative.
        # Rank by image alignment (NCC + small NMI term) over common FOV.
        fv_empty_mask = float(fv.sum()) < 1.0
        use_img_score = (any_mask_missing or fv_empty_mask) and fi is not None and mi is not None
        if use_img_score:
            sp_scored = []
            for lbl, M, d in sp:
                img_score = _image_alignment_score(fi, mi, M, H, W)
                sp_scored.append((lbl, M, d, img_score))
                print(f"  [SparseImgScore] {lbl:<26} score={img_score:.3f}")
        else:
            sp_scored = [(lbl, M, d, anatomical_score(M, fv, mv, ap)) for lbl, M, d in sp]
        if use_img_score or no_asd_selection:
            # When fv is empty, asd_reward is meaningless — pick directly by
            # image alignment score (t_score column) instead of _final_select.
            best_lbl, best_M, best_d, best_t = max(sp_scored, key=lambda x: x[3])
            sig, wr = "image_score" if use_img_score else "anatomical", None
            # Optional fallback to scale_best for non-missing-mask cases only.
            if use_img_score and (not any_mask_missing) and "LoFTR" in best_lbl:
                scale_best_scores = [s for l, _, _, s in sp_scored if "scale_best" in l]
                if scale_best_scores:
                    sb_score = scale_best_scores[0]
                    # Use scale_best if: (1) LoFTR score is very low, or (2) scale_best is close
                    if best_t < 0.25 or (best_t - sb_score) < 0.05:
                        print(f"  [SparseImgScore] LoFTR score={best_t:.3f} vs scale_best={sb_score:.3f}; using scale_best")
                        scale_best_entry = [e for e in sp_scored if "scale_best" in e[0]][0]
                        best_lbl, best_M, best_d, best_t = scale_best_entry
                        sig = "image_score→scale_best_fallback"
        else:
            (best_lbl, best_M, best_d, best_t), sig, wr = _final_select(sp_scored, fv, mv, ap)
        oracle_pool = ([(l, Mx) for l, Mx, *_ in sp_scored], best_lbl)
        print(f"  [Sparse] final pick by {sig}"
              + (f" (ASD-reward={wr:.3f})" if wr is not None else "")
              + f"  anat={best_d:.3f}  [{best_lbl}]")
        pre_flow_M, pre_flow_d, pre_flow_lbl = best_M, best_d, best_lbl
        if not no_multires_flow:
            M_fb, d_fb = tvl1_flow_to_affine_multires(fv, mv, best_M, ap, bounds=bounds)
            a_best = anatomical_score(best_M, fv, mv, ap)
            a_fb   = anatomical_score(M_fb, fv, mv, ap) if M_fb is not None else -1.0
            if utils.sane(M_fb, ap, bounds) and a_fb > a_best:
                best_M, best_d, best_lbl = M_fb, d_fb, best_lbl + "+flow"
        if not no_cascade:
            # Auto-detect low-quality FAF mask for this moving image and treat
            # such cases as aggressive (skip cascade / prefer pre-flow).
            try:
                mv_bin_local = (mv > 0).astype('uint8')
                n_labels_local, labels_local, stats_local, _ = cv2.connectedComponentsWithStats(mv_bin_local, connectivity=8)
                if n_labels_local <= 1:
                    largest_cc_ratio_local = 0.0
                else:
                    areas_local = stats_local[1:, cv2.CC_STAT_AREA]
                    total_area_local = float(areas_local.sum()) if areas_local.size else 0.0
                    largest_cc_ratio_local = float(areas_local.max()) / max(1.0, total_area_local) if areas_local.size else 0.0
            except Exception:
                largest_cc_ratio_local = 0.0

            mask_low_quality_local = (float(mv.sum()) < 50) or (largest_cc_ratio_local < 0.05)
            current_row = getattr(utils, "CURRENT_ROW", None)
            aggressive_row = (
                mask_low_quality_local
                or (current_row is not None and current_row in getattr(utils, "CASCADE_AGGRESSIVE_ROWS", set()))
            )
            if aggressive_row:
                print(f"  [CascadeGuard] forced pre-cascade for row {current_row} (mask_low_quality={mask_low_quality_local})")
                best_M, best_d, best_lbl = pre_flow_M, pre_flow_d, pre_flow_lbl + "|precascade"
            else:
                best_M, best_d, best_lbl = cascade_refine(
                    fv, mv, best_M, best_lbl, ap,
                    no_multires_flow=no_multires_flow, no_comp_flow=no_comp_flow,
                    bounds=bounds)
        print(f"  [Sparse] winner={best_lbl}  anat={best_d:.3f}")
        scale_meta["winner_scale"] = utils.parse_winner_scale(best_lbl, best_s)
        return best_lbl, best_M, best_d, scale_meta, oracle_pool

    candidates = []
    seed_scale = float(utils.FOV_SCALE_RATIO)

    def add(lbl, M, d_skel=None):
        if not utils.sane(M, ap, bounds): return
        if d_skel is None: d_skel = anatomical_score(M, fv, mv, ap)
        t_score = anatomical_score(M, fv, mv, ap)
        candidates.append((lbl, M, d_skel, t_score))
        print(f"  [Cand] {lbl:<30} anat={d_skel:.3f}  topo={t_score:.3f}")

    add("scale_best", M_scale_best, d_scale_best)
    M_disc, disc_meta = _disc_anchor_candidate(fi, mi, M_scale_best, H, W)
    if M_disc is not None and utils.sane(M_disc, ap, bounds):
        add("disc_anchor", M_disc)
        print("  [DiscAnchor] "
              f"fixed=({disc_meta['fixed_center'][0]:.1f},{disc_meta['fixed_center'][1]:.1f}) "
              f"moving=({disc_meta['moving_center'][0]:.1f},{disc_meta['moving_center'][1]:.1f}) "
              f"shift=({disc_meta['shift'][0]:+.1f},{disc_meta['shift'][1]:+.1f})")
    if utils.MULTISCALE_LOFTR:
        cx, cy = W / 2.0, H / 2.0
        scales = utils.multiscale_grid(best_s)
        print(f"  [MultiScale] grid {scales}")
        loftr_starts = []
        for s in scales:
            M_s = utils.make_scale_matrix(float(s), cx, cy)
            add(f"scale@{s:.2f}", M_s)            # seed candidate per scale
            loftr_starts.append((M_s, f"{s:.2f}"))
    else:
        loftr_starts = [(M_scale_best, f"{best_s:.2f}")]
        if M_disc is not None and utils.sane(M_disc, ap, bounds):
            loftr_starts.append((M_disc, "disc"))
        if utils.SCALE_SEARCH_ENABLED and abs(seed_scale - best_s) > 1e-6:
            M_seed = utils.make_scale_matrix(seed_scale, W / 2.0, H / 2.0)
            if utils.sane(M_seed, ap, bounds):
                add("scale_seed", M_seed)
                loftr_starts.append((M_seed, f"{seed_scale:.2f}"))
                print(f"  [ScaleSearch] also keeping FOV seed {seed_scale:.3f} as a LoFTR/flow start")
    if not no_loftr:
        for M_start, start_lbl in loftr_starts:
            for lbl, M, d in _loftr_from_scale(fv, mv, M_start, start_lbl, H, W, ap,
                                               no_multires_flow=no_multires_flow,
                                               bounds=bounds,
                                               fi=fi, mi=mi,
                                               any_mask_missing=any_mask_missing):
                add(lbl, M, d)

    if not candidates:
        print(f"  [Cand] WARNING — all candidates failed sanity; "
              f"force-adding scale_best as fallback")
        d_fb_scale = anatomical_score(M_scale_best, fv, mv, ap)
        t_fb_scale = anatomical_score(M_scale_best, fv, mv, ap)
        candidates.append(("scale_best(forced)", M_scale_best,
                           d_fb_scale, t_fb_scale))

    best_so_far = max(candidates, key=lambda x: x[3])
    print(f"  [Cand] best-anat before flow: {best_so_far[0]} = {best_so_far[3]:.3f}")
    loftr_won = any("LoFTR" in c[0] for c in candidates
                    if c[3] >= best_so_far[3] - 0.01)
    if not loftr_won and not no_multires_flow:
        M_fb, d_fb = tvl1_flow_to_affine_multires(fv, mv, best_so_far[1], ap, bounds=bounds)
        if utils.sane(M_fb, ap, bounds): add("flow->affine", M_fb, d_fb)

    # When every vessel candidate is weak but image LoFTR recovered a viable
    # transform, vessel ASD is no longer trustworthy for final selection.
    # Select the image candidate by cross-modal NMI instead.
    best_vessel_score = max(
        (c[3] for c in candidates if "LoFTR-img" not in c[0]), default=-1.0)
    image_candidates = [c for c in candidates if "LoFTR-img" in c[0]]
    use_image_fallback = (
        bool(image_candidates)
        and best_vessel_score < utils.IMAGE_FALLBACK_MIN_ANATOMICAL
        and fi is not None and mi is not None
    )
    if use_image_fallback:
        best_lbl, best_M, best_d, best_t = max(
            image_candidates,
            key=lambda c: _image_alignment_score(fi, mi, c[1], H, W),
        )
        sig, wr = "image_score(vessel_fallback)", None
    elif no_asd_selection:
        best_lbl, best_M, best_d, best_t = max(candidates, key=lambda x: x[3])
        sig, wr = "anatomical", None
    else:
        (best_lbl, best_M, best_d, best_t), sig, wr = _final_select(candidates, fv, mv, ap)
    oracle_pool = ([(l, Mx) for l, Mx, *_ in candidates], best_lbl)
    scale_meta["winner_scale"] = utils.parse_winner_scale(best_lbl, best_s)
    print(f"  [FinalPick] by {sig}"
          + (f" (ASD-reward={wr:.3f})" if wr is not None else "")
          + f"  -> {best_lbl}")
    print(f"  [Winner pre-cascade] {best_lbl}  anat={best_d:.3f}  topo={best_t:.3f}")
    pre_flow_M, pre_flow_d, pre_flow_lbl = best_M, best_d, best_lbl
    if not no_cascade:
        # Auto-detect low-quality FAF mask for this moving image and treat
        # such cases as aggressive (skip cascade / prefer pre-flow).
        try:
            mv_bin_local = (mv > 0).astype('uint8')
            n_labels_local, labels_local, stats_local, _ = cv2.connectedComponentsWithStats(mv_bin_local, connectivity=8)
            if n_labels_local <= 1:
                largest_cc_ratio_local = 0.0
            else:
                areas_local = stats_local[1:, cv2.CC_STAT_AREA]
                total_area_local = float(areas_local.sum()) if areas_local.size else 0.0
                largest_cc_ratio_local = float(areas_local.max()) / max(1.0, total_area_local) if areas_local.size else 0.0
        except Exception:
            largest_cc_ratio_local = 0.0

        mask_low_quality_local = (float(mv.sum()) < 50) or (largest_cc_ratio_local < 0.05)
        current_row = getattr(utils, "CURRENT_ROW", None)
        aggressive_row = (
            mask_low_quality_local
            or (current_row is not None and current_row in getattr(utils, "CASCADE_AGGRESSIVE_ROWS", set()))
        )
        if aggressive_row:
            print(f"  [CascadeGuard] forced pre-cascade for row {current_row} (mask_low_quality={mask_low_quality_local})")
            best_M, best_d, best_lbl = pre_flow_M, pre_flow_d, pre_flow_lbl + "|precascade"
        else:
            best_M, best_d, best_lbl = cascade_refine(
                fv, mv, best_M, best_lbl, ap,
                no_multires_flow=no_multires_flow, no_comp_flow=no_comp_flow,
                bounds=bounds)
    return best_lbl, best_M, best_d, scale_meta, oracle_pool