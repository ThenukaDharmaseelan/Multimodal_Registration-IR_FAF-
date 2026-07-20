"""fafir_registration.pipeline — the single user-facing entry point.

Wraps the whole modular pipeline behind one call::

    from fafir_registration import register

    result = register(
        ir_image=ir,          # IR (30°) — fixed
        faf_image=faf,        # FAF (55°) — moving
        ir_vessel=ir_mask,
        faf_vessel=faf_mask,
    )
    result.transform          # 2x3 affine mapping moving -> fixed
    result.registered_image   # FAF warped onto the IR frame (RGB)
    result.metrics            # before/after quality metrics

Inputs may be numpy arrays or image paths. Everything below the call — FOV
scaling, LoFTR, optical flow, anatomical scoring, candidate selection — is
handled by the other modules and never exposed to the caller, but each stage
can be switched off individually (``use_loftr``, ``multires_flow``,
``comp_flow``, ``cascade``, ``asd_selection``, ``scale_search``,
``preprocess``) for ablation studies.

Flagging is post-registration only: every pair is registered fully, then
flagged (``status="flagged"``) by one of three checks — a noisy image
(``flag_noise_over``), a sparse/unsegmentable vessel mask (``flag_sparse``),
or a low final anatomical score (``flag_min_anat``). Registration/matching is
never altered — flagging only decides whether a registered result is kept or
set aside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from . import metrics as M
from . import utils, visualization
from .io import load_image, load_vessel
from .models import loftr
from .preprocessing import (
    _image_visibility_metrics,
    assess_vessel_mask,
    estimate_vessel_mask,
    flag_vessel_pair,
    preprocess_image,
)
from .registration import register_multi_candidate

ArrayOrPath = Union[np.ndarray, str, Path]

__all__ = ["register", "RegistrationResult"]


# ─────────────────────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistrationResult:
    """Everything a caller needs from one registration, and nothing they don't.

    Attributes
    ----------
    transform : np.ndarray
        Final 2x3 affine mapping the moving (FAF) image onto the fixed (IR)
        frame. Apply with ``cv2.warpAffine(img, transform, (W, H))``.
    registered_image : np.ndarray
        Moving image warped into the fixed frame (RGB, uint8).
    registered_vessel : np.ndarray
        Moving vessel mask warped into the fixed frame (float32, 0/1).
    metrics : dict
        Before/after registration-quality metrics.
    overlay, checkerboard, vessel_overlap, vessel_mismatch : np.ndarray
        Diagnostic views.
    label : str
        Which internal candidate won (e.g. ``"LoFTR@1.83+flow+comp"``).
    scale : float
        The effective FOV scale used.
    quality : str
        Vessel-presence tier or flag verdict.
    status : str
        ``"ok"`` for a normal registration, ``"flagged"`` when rejected.
    flag_verdict, flag_reason : str
        The verdict (e.g. ``"sparse"``/``"noisy"``) and reason when flagged.
    """

    transform: np.ndarray
    registered_image: np.ndarray
    registered_vessel: np.ndarray
    metrics: dict = field(default_factory=dict)
    overlay: np.ndarray = None
    checkerboard: np.ndarray = None
    vessel_overlap: np.ndarray = None
    vessel_mismatch: np.ndarray = None
    label: str = ""
    scale: float = 0.0
    quality: str = ""
    status: str = "ok"
    flag_verdict: str = ""
    flag_reason: str = ""
    # Loaded/resized inputs kept for convenience (e.g. saving grids).
    fixed_image: np.ndarray = None
    moving_image: np.ndarray = None
    fixed_vessel: np.ndarray = None
    moving_vessel: np.ndarray = None

    def save(self, output_dir: Union[str, Path], name: str = "pair") -> Path:
        """Write the full diagnostic image set (registered image, overlays,
        checkerboard, vessel views and the annotated grid) under
        ``output_dir``. Flagged results are rendered with the same full grid,
        with the flag verdict shown in the sidebar stage label."""
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        H, W = self.fixed_image.shape[:2]
        M_scale = utils.make_scale_matrix(self.scale, W / 2.0, H / 2.0)
        decomp = utils.decompose_affine(self.transform)
        _stage = (self.label if self.status == "ok"
                  else f"FLAGGED:{self.flag_verdict}")
        _args = (od, name,
                 self.fixed_image, self.fixed_vessel,
                 self.moving_image, self.moving_vessel,
                 self.registered_image, self.registered_vessel,
                 M_scale)
        _kw = dict(final_transform=self.transform, decomp=decomp,
                   stage=_stage, best_scale=self.scale,
                   scale_cluster="fov", quality_label=self.quality)
        try:
            # Preferred: pass quality_label so the grid shows IQuality:<tier>.
            visualization.save_outputs(*_args, **_kw)
        except TypeError:
            # Older visualization.save_outputs without quality_label.
            _kw.pop("quality_label", None)
            visualization.save_outputs(*_args, **_kw)
        return od


# ─────────────────────────────────────────────────────────────────────────────
# Input loading
# ─────────────────────────────────────────────────────────────────────────────

def _as_image(x: ArrayOrPath, h: int, w: int) -> np.ndarray:
    """Return an (h, w, 3) uint8 RGB image from an array or a path."""
    if isinstance(x, (str, Path)):
        return load_image(x, h, w)
    arr = np.asarray(x)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr] * 3, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr


def _vessel_mask_missing(x) -> bool:
    """Return whether a CSV/path vessel-mask value is intentionally missing."""
    if x is None:
        return True
    if isinstance(x, str):
        return not x.strip() or not Path(x).exists()
    if isinstance(x, Path):
        return not x.exists()
    # Pandas represents blank CSV cells as float NaN.
    return isinstance(x, (float, np.floating)) and bool(np.isnan(x))


def _as_vessel(x: Optional[ArrayOrPath], h: int, w: int) -> np.ndarray:
    """Return an (h, w) float32 {0,1} vessel mask from an array or a path."""
    if _vessel_mask_missing(x):
        print("  [WARN] missing vessel mask")
        return np.zeros((h, w), dtype=np.float32)
    if isinstance(x, (str, Path)):
        return load_vessel(x, h, w)
    arr = np.asarray(x)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    thr = 0.5 if arr.max() <= 1.0 else 127
    return (arr > thr).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics assembly
# ─────────────────────────────────────────────────────────────────────────────

def _collect_metrics(fi, fv, mi, mv, wi, wv, transform, scale) -> dict:
    """Before/after quality metrics for one registered pair."""
    H, W = fv.shape

    dsc_before = round(M.dice(fv, mv), 4)
    dsc_after  = round(M.dice(fv, wv), 4)
    ncc_before = round(M.ncc(fv, mv), 4)
    ncc_after  = round(M.ncc(fv, wv), 4)

    hd_before  = M.hausdorff_vessel_distance(fv, mv)
    topo       = M.compute_all_topology_metrics(transform, fv, mv)
    wass_before = M.wasserstein_vessel_distance(fv, mv)
    ssd_before = M.ssd_vessel_distance(fv, mv)
    ssd_after  = M.ssd_vessel_distance(fv, wv)

    cl_before   = M.centerline_overlap_score(fv, mv).get("cl_recall_mean")
    conn_before = M.vessel_connectivity_score(fv, mv).get("conn_cc_score")

    # image-intensity (greyscale, inside common FOV)
    M_scale = utils.make_scale_matrix(scale, W / 2.0, H / 2.0)
    mi_scaled = utils.warp_img(mi, M_scale, H, W)
    fov_before = ((utils.get_fov_mask(fi) > 0) & (utils.get_fov_mask(mi_scaled) > 0)).astype(np.float32)
    fov_after  = ((utils.get_fov_mask(fi) > 0) & (utils.get_fov_mask(wi) > 0)).astype(np.float32)
    img_before = M.image_intensity_metrics(fi, mi_scaled, fov_mask=fov_before)
    img_after  = M.image_intensity_metrics(fi, wi, fov_mask=fov_after)

    def _image_binary_dice(a_rgb, b_rgb, fov_mask):
        """Dice on binarized grayscale images inside a common FOV mask."""
        if fov_mask is None or float(fov_mask.sum()) < 1.0:
            return None
        a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        m = fov_mask > 0
        if int(m.sum()) < 100:
            return None

        # Zero out outside-FOV so thresholding is driven by valid retina area.
        a_m = a.copy(); a_m[~m] = 0
        b_m = b.copy(); b_m[~m] = 0
        _, a_bin = cv2.threshold(a_m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b_bin = cv2.threshold(b_m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        aa = (a_bin > 0) & m
        bb = (b_bin > 0) & m
        denom = float(aa.sum() + bb.sum())
        if denom < 1.0:
            return None
        return round(float(2.0 * (aa & bb).sum() / denom), 4)

    image_dice_before = _image_binary_dice(fi, mi_scaled, fov_before)
    image_dice_after = _image_binary_dice(fi, wi, fov_after)
    image_ncc_before = img_before.get("ncc_img")
    image_ncc_after = img_after.get("ncc_img")

    def _image_ssd(a_rgb, b_rgb, fov_mask):
        if fov_mask is None:
            return None
        m = fov_mask > 0
        if int(m.sum()) < 16:
            return None
        a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        ssd = float(np.sum((a[m] - b[m]) ** 2))
        return round(ssd, 4)

    image_ssd_before = _image_ssd(fi, mi_scaled, fov_before)
    image_ssd_after = _image_ssd(fi, wi, fov_after)

    vessel_ssd_before = ssd_before["ssd"]
    vessel_ssd_after = ssd_after["ssd"]

    def _d(b, a, higher=True):
        if b is None or a is None:
            return None
        return round((a - b) if higher else (b - a), 4)

    wass_after = topo.get("wass_w2") if topo.get("wass_w2") is not None else topo.get("wass_w1")
    wass_bef   = (wass_before.get("wass_w2") if wass_before.get("wass_w2") is not None
                  else wass_before.get("wass_w1"))

    return dict(
        # --- explicit vessel metrics requested ---
        vessel_dice_before=dsc_before,
        vessel_dice_after=dsc_after,
        vessel_dice_delta=_d(dsc_before, dsc_after, higher=True),
        vessel_ncc_before=ncc_before,
        vessel_ncc_after=ncc_after,
        vessel_ncc_delta=_d(ncc_before, ncc_after, higher=True),
        vessel_ssd_before=vessel_ssd_before,
        vessel_ssd_after=vessel_ssd_after,
        vessel_ssd_delta=_d(vessel_ssd_before, vessel_ssd_after, higher=False),
        # --- explicit image metrics requested ---
        image_dice_before=image_dice_before,
        image_dice_after=image_dice_after,
        image_dice_delta=_d(image_dice_before, image_dice_after, higher=True),
        image_ncc_before=image_ncc_before,
        image_ncc_after=image_ncc_after,
        image_ncc_delta=_d(image_ncc_before, image_ncc_after, higher=True),
        image_ssd_before=image_ssd_before,
        image_ssd_after=image_ssd_after,
        image_ssd_delta=_d(image_ssd_before, image_ssd_after, higher=False),
        # --- vessel-mask overlap (higher = better) ---
        dice_before=dsc_before, dice_after=dsc_after,
        dice_delta=_d(dsc_before, dsc_after, higher=True),
        ncc_before=ncc_before, ncc_after=ncc_after,
        ncc_delta=_d(ncc_before, ncc_after, higher=True),
        centerline_recall_before=cl_before,
        centerline_recall_after=topo.get("cl_recall_mean"),
        connectivity_before=conn_before,
        connectivity_after=topo.get("conn_cc_score"),
        # --- surface / distribution distances (lower = better) ---
        hd95_before=hd_before.get("hd95"), hd95_after=topo.get("hd95"),
        hd95_delta=_d(hd_before.get("hd95"), topo.get("hd95"), higher=False),
        asd_before=hd_before.get("masd"), asd_after=topo.get("masd"),
        asd_delta=_d(hd_before.get("masd"), topo.get("masd"), higher=False),
        wasserstein_before=wass_bef, wasserstein_after=wass_after,
        wasserstein_delta=_d(wass_bef, wass_after, higher=False),
        ssd_before=vessel_ssd_before, ssd_after=vessel_ssd_after,
        ssd_delta=_d(vessel_ssd_before, vessel_ssd_after, higher=False),
        # --- image intensity (cross-modal; MI/NMI primary) ---
        mi_before=img_before["mi"], mi_after=img_after["mi"],
        nmi_before=img_before["nmi"], nmi_after=img_after["nmi"],
        ssim_before=img_before["ssim"], ssim_after=img_after["ssim"],
        psnr_before=img_before["psnr"], psnr_after=img_after["psnr"],
        # --- transform decomposition ---
        **{f"transform_{k}": round(v, 4)
           for k, v in utils.decompose_affine(transform).items()},
    )


def _flagged_result(verdict, reason, *, transform, wi, wv, scale, label,
                    fi_raw, mi_raw, fv, mv):
    """Build a flagged RegistrationResult (status='flagged')."""
    return RegistrationResult(
        transform=transform,
        registered_image=wi,
        registered_vessel=wv,
        metrics={},
        label=label,
        scale=scale,
        quality=verdict,
        status="flagged",
        flag_verdict=verdict,
        flag_reason=reason,
        fixed_image=fi_raw, moving_image=mi_raw,
        fixed_vessel=fv, moving_vessel=mv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The one public entry point
# ─────────────────────────────────────────────────────────────────────────────

def register(
    ir_image: Optional[ArrayOrPath] = None,
    faf_image: Optional[ArrayOrPath] = None,
    ir_vessel: Optional[ArrayOrPath] = None,
    faf_vessel: Optional[ArrayOrPath] = None,
    *,
    fov_scale: Optional[float] = None,
    size: int = 224,
    preprocess: bool = True,
    use_loftr: bool = True,
    scale_search: bool = True,
    multires_flow: bool = True,
    comp_flow: bool = True,
    cascade: bool = True,
    asd_selection: bool = True,
    adaptive_sanity: Optional[bool] = None,
    compute_metrics: bool = True,
    # ── flagging: post-registration only ──
    flag_min_anat: Optional[float] = None,
    flag_sparse: bool = False,
    flag_unsegmentable: bool = False,
    sparse_thresholds: Optional[dict] = None,
    flag_noise_over: Optional[float] = None,
    flag_noise_keep_if_segmentable: bool = True,
    fixed_image: Optional[ArrayOrPath] = None,
    moving_image: Optional[ArrayOrPath] = None,
    fixed_vessel: Optional[ArrayOrPath] = None,
    moving_vessel: Optional[ArrayOrPath] = None,
) -> RegistrationResult:
    """Register a moving (FAF) image onto a fixed (IR) image.

    Parameters
    ----------
    ir_image, faf_image, ir_vessel, faf_vessel
        numpy arrays or paths. IR = fixed (30°), FAF = moving (55°).
    fov_scale, size, preprocess, use_loftr, scale_search, multires_flow,
    comp_flow, cascade, asd_selection, adaptive_sanity, compute_metrics
        Engine knobs (see module docstring).

    flag_noise_over
        Post-registration IMAGE gate: flag (verdict ``"noisy"``) when an image's
        Immerkaer noise estimate exceeds this value (e.g. 16). None/0 disables it.
    flag_noise_keep_if_segmentable
        When True (default), a noisy image is NOT flagged if both vessel masks
        are well-segmented (``"ok"``) — registration is mask-driven, so grain
        doesn't matter. Set False to flag on noise regardless of mask quality.
    flag_min_anat
        Post-registration: flag (verdict ``"low_quality"``) when the winning
        candidate's final anatomical score is below this value. None disables it.
    flag_sparse
        Post-registration: flag (verdict ``"sparse"`` / ``"unsegmentable"``)
        when a vessel mask is thin/sparse OR dots/speckle. Off by default.
    flag_unsegmentable
        Post-registration: flag ONLY the ``"unsegmentable"`` case — dots/speckle
        with no coherent vessel tree. Decent thin (``"sparse"``) masks are kept.
        Use this to catch noise/speckle masks without flagging usable thin ones.
        Off by default.

    In every case matching is never altered by flagging — a flag only decides
    whether the registered result is kept or set aside.

    Returns
    -------
    RegistrationResult
    """
    if size % 14 != 0:
        raise ValueError(f"size must be a multiple of 14, got {size}.")

    if ir_image is None:
        ir_image = fixed_image
    if faf_image is None:
        faf_image = moving_image
    if ir_vessel is None:
        ir_vessel = fixed_vessel
    if faf_vessel is None:
        faf_vessel = moving_vessel

    if ir_image is None or faf_image is None:
        raise TypeError("register() requires ir_image and faf_image")

    # Configure the (module-global) engine knobs for this call.
    if fov_scale is not None:
        utils.FOV_SCALE_RATIO = float(fov_scale)
    utils.SCALE_SEARCH_ENABLED = bool(scale_search)
    if adaptive_sanity is not None:
        utils.ADAPTIVE_SANITY = bool(adaptive_sanity)

    h = w = size

    fi = _as_image(ir_image, h, w)
    mi = _as_image(faf_image, h, w)

    # When a segmentation file is unavailable, derive a provisional vessel
    # mask from the image so vessel registration can still be tried.
    _ir_vessel_missing = _vessel_mask_missing(ir_vessel)
    _faf_vessel_missing = _vessel_mask_missing(faf_vessel)
    fv = estimate_vessel_mask(fi) if _ir_vessel_missing else _as_vessel(ir_vessel, h, w)
    mv = estimate_vessel_mask(mi) if _faf_vessel_missing else _as_vessel(faf_vessel, h, w)
    if _ir_vessel_missing:
        print(f"  [PseudoVessel] IR image-derived pixels={int(fv.sum())}")
    if _faf_vessel_missing:
        print(f"  [PseudoVessel] FAF image-derived pixels={int(mv.sum())}")

    fi_raw = fi.copy()
    mi_raw = mi.copy()

    # No pre-registration quality *tier*, but preserve the original routing:
    # a mask with no coherent vessel tree (empty OR dots/speckle) is sent to
    # image-based feature matching, exactly as classify_pair_quality's
    # "no_vessels" did. This keeps registration unchanged for those pairs.
    any_mask_missing = (assess_vessel_mask(fv)[0] == "unsegmentable"
                        or assess_vessel_mask(mv)[0] == "unsegmentable")
    quality = ""

    if preprocess:
        kw = dict(clip_limit=8.0, tile_grid=(4, 4), blur_ksize=3, blur_sigma=0.5)
        fi = preprocess_image(fi, **kw)
        mi = preprocess_image(mi, **kw)

    if use_loftr and not loftr.LOFTR_AVAILABLE:
        loftr.init_loftr()

    label, transform, _anat, scale_meta, _oracle = register_multi_candidate(
        fv, mv, h, w,
        no_loftr=not use_loftr,
        no_multires_flow=not multires_flow,
        no_comp_flow=not comp_flow,
        no_asd_selection=not asd_selection,
        no_cascade=not cascade,
        fi=fi, mi=mi,
        any_mask_missing=any_mask_missing,
    )

    scale = float(scale_meta.get("winner_scale")
                  or scale_meta.get("scale_bifurc_est")
                  or utils.FOV_SCALE_RATIO)

    wi = utils.warp_img(mi_raw, transform, h, w)
    wv = utils.warp_mask(mv, transform, h, w)

    # ── post-registration flags (pair is fully registered before judging) ────
    # (a) noisy IMAGE — flag if the Immerkaer noise estimate is high. A noisy
    #     image whose vessel masks are still well-segmented is kept, since
    #     registration is mask-driven.
    if flag_noise_over is not None and float(flag_noise_over) > 0:
        _ir_noise = float(_image_visibility_metrics(fi_raw)["noise"])
        _faf_noise = float(_image_visibility_metrics(mi_raw)["noise"])
        if max(_ir_noise, _faf_noise) > float(flag_noise_over):
            _masks_good = False
            if flag_noise_keep_if_segmentable:
                _masks_good = (assess_vessel_mask(fv)[0] == "ok"
                               and assess_vessel_mask(mv)[0] == "ok")
            if _masks_good:
                print(f"  [NoiseGate] noise high "
                      f"(IR={_ir_noise:.1f},FAF={_faf_noise:.1f}"
                      f">{float(flag_noise_over):.1f}) but masks segmentable — kept")
            else:
                _reason = (f"noise(IR={_ir_noise:.1f},FAF={_faf_noise:.1f}"
                           f">{float(flag_noise_over):.1f})")
                print(f"  [FLAGGED] noisy: {_reason}")
                return _flagged_result("noisy", _reason,
                                       transform=transform, wi=wi, wv=wv, scale=scale,
                                       label=label, fi_raw=fi_raw, mi_raw=mi_raw, fv=fv, mv=mv)

    # (b) vessel-mask verdicts. --flag-unsegmentable fires ONLY on dots/speckle
    #     (no coherent tree); --flag-sparse additionally fires on thin-but-real
    #     masks. A pair with a decent-thin (sparse) FAF mask is kept unless
    #     --flag-sparse is passed.
    if flag_sparse or flag_unsegmentable:
        _sflag, _sverdict, _sreason, _ = flag_vessel_pair(
            fv, mv, flag_sparse=flag_sparse, **(sparse_thresholds or {}))
        _hit = ((flag_sparse and _sverdict in ("sparse", "unsegmentable"))
                or (flag_unsegmentable and _sverdict == "unsegmentable"))
        if _hit:
            print(f"  [FLAGGED] {_sverdict}: {_sreason}")
            return _flagged_result(_sverdict, _sreason,
                                   transform=transform, wi=wi, wv=wv, scale=scale,
                                   label=label, fi_raw=fi_raw, mi_raw=mi_raw, fv=fv, mv=mv)

    # (c) low registration quality — non-sparse pairs that registered badly.
    if flag_min_anat is not None and float(_anat) < float(flag_min_anat):
        _reason = f"anat={float(_anat):.3f}<{float(flag_min_anat):.3f}(winner={label})"
        print(f"  [FLAGGED] low_quality: {_reason}")
        return _flagged_result("low_quality", _reason,
                               transform=transform, wi=wi, wv=wv, scale=scale,
                               label=label, fi_raw=fi_raw, mi_raw=mi_raw, fv=fv, mv=mv)

    metrics = (_collect_metrics(fi_raw, fv, mi_raw, mv, wi, wv, transform, scale)
               if compute_metrics else {})

    return RegistrationResult(
        transform=transform,
        registered_image=wi,
        registered_vessel=wv,
        metrics=metrics,
        overlay=visualization.make_overlap_image(fi_raw, wi),
        checkerboard=visualization.make_checkerboard(fi_raw, wi),
        vessel_overlap=visualization.make_overlap_vessels(fv, wv, fi_raw),
        vessel_mismatch=visualization.make_vessel_mismatch(fv, wv, fi_raw),
        label=label,
        scale=scale,
        quality=quality,
        status="ok",
        fixed_image=fi_raw, moving_image=mi_raw,
        fixed_vessel=fv, moving_vessel=mv,
    )