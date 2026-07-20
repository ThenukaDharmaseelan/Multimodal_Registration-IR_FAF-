"""fafir.cli — batch registration (and ablation study) from the command line.

Normal batch run::

    fafir-register pairs.csv output/

Ablation study (leave-one-out over the pipeline components)::

    fafir-register pairs.csv output/ --ablation
    fafir-register pairs.csv output/ --ablation --limit 5
    fafir-register pairs.csv output/ --ablation --configs full no_loftr

``pairs.csv`` lists one image pair per row. By default the columns are:

    fixed               IR image path
    moving              FAF image path
    fixed_vessel_mask   IR vessel-mask path
    moving_vessel_mask  FAF vessel-mask path

(column names are configurable via flags).

Normal mode registers each pair, writes the diagnostic image set into
``output/``, appends per-pair metrics to ``output/results.csv`` and prints
mean ± std summaries (Dice, HD95, NCC, Wasserstein, ASD — before/after/delta),
overall (also saved to ``output/summary.csv``).

Flagging (all opt-in): pairs that can't be usefully registered are flagged and
written to ``output/flagged/`` with the full diagnostic grid, and recorded with
``status=flagged`` in results.csv (kept out of the summaries). No flag triggers
image-based registration — flagged pairs are simply rejected.

Ablation mode instead runs every pair once per configuration (full pipeline,
then each component disabled in turn) and writes:

    output/ablation_results.csv   one row per (pair x config)
    output/ablation_summary.csv   mean ± std per (config x metric)
    output/ablation_table.md      paper-ready ✓/✗ markdown table
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils
from .io import load_image, load_vessel
from .pipeline import register


def _seed_everything(seed: int = 0) -> None:
    """Seed all RNGs so a run is reproducible (same transform every time):
    NumPy/Python, OpenCV RANSAC, and Torch/cuDNN for LoFTR."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2
        cv2.setRNGSeed(seed)          # OpenCV RANSAC (estimateAffine*)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True   # deterministic LoFTR convs
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
from .preprocessing import (
    classify_pair_quality,
    classify_pair_readability,
    estimate_optic_disc_center,
)

SUMMARY_METRICS = ["dice", "hd95", "ncc", "wasserstein", "asd"]
# (display_name, preferred_stem, fallback_stem)
SUMMARY_METRICS_VESSEL = [
    ("dice", "vessel_dice", "dice"),
    ("ncc", "vessel_ncc", "ncc"),
    ("ssd", "vessel_ssd", "ssd"),
    ("hd95", "vessel_hd95", "hd95"),
    ("asd", "vessel_asd", "asd"),
    ("wasserstein", "vessel_wasserstein", "wasserstein"),
    ("centerline_recall", "vessel_centerline_recall", "centerline_recall"),
    ("connectivity", "vessel_connectivity", "connectivity"),
]
SUMMARY_METRICS_IMAGE = [
    ("dice", "image_dice", None),
    ("ncc", "image_ncc", None),
    ("ssd", "image_ssd", None),
    ("mi", "image_mi", "mi"),
    ("nmi", "image_nmi", "nmi"),
    ("ssim", "image_ssim", "ssim"),
    ("psnr", "image_psnr", "psnr"),
]


def _as_float(v):
    try:
        return float(v)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ablation matrix (leave-one-out). Overrides are kwargs of ``register``.
# ─────────────────────────────────────────────────────────────────────────────

ABLATIONS = [
    # name                 pretty label              register() overrides
    ("full",              "Full method",             {}),
    ("no_loftr",          "w/o LoFTR",               dict(use_loftr=False)),
    ("no_optical_flow",   "w/o Optical Flow",        dict(multires_flow=False,
                                                          comp_flow=False)),
    ("no_cascade",        "w/o Cascade",             dict(cascade=False)),
    ("no_asd_selection",  "w/o ASD Selection",       dict(asd_selection=False)),
    ("no_scale_search",   "Fixed FOV (no search)",   dict(scale_search=False)),
    ("no_preprocess",     "w/o CLAHE Preprocess",    dict(preprocess=False)),
    ("seed_only",         "FOV Seed only",           dict(use_loftr=False,
                                                          scale_search=False,
                                                          multires_flow=False,
                                                          comp_flow=False,
                                                          cascade=False,
                                                          asd_selection=False)),
]

# Component columns for the ✓/✗ table: (header, register-kwarg)
COMPONENTS = [
    ("Scale Search", "scale_search"),
    ("CLAHE",        "preprocess"),
    ("LoFTR",        "use_loftr"),
    ("Opt. Flow",    "multires_flow"),
    ("Cascade",      "cascade"),
    ("ASD Sel.",     "asd_selection"),
]

# Ablation-table metrics: (column stem, header, higher_is_better)
ABL_METRICS = [
    ("dice",        "Dice ↑",  True),
    ("hd95",        "HD95 ↓",  False),
    ("asd",         "ASD ↓",   False),
    ("ncc",         "NCC ↑",   True),
    ("wasserstein", "Wass ↓",  False),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fafir-register",
        description="Batch IR<->FAF registration over a CSV of image pairs.")
    p.add_argument("csv", help="CSV listing image pairs (one per row).")
    p.add_argument("output", help="Output directory for images + results.csv.")
    p.add_argument("--ir-col", "--fixed-col", dest="ir_col", default="moving",
                   help="CSV column with the IR image path.")
    p.add_argument("--faf-col", "--moving-col", dest="faf_col", default="fixed",
                   help="CSV column with the FAF image path.")
    p.add_argument("--ir-vessel-col", "--fixed-vessel-col", dest="ir_vessel_col", default="moving_vessel_mask",
                   help="CSV column with the IR vessel-mask path.")
    p.add_argument("--faf-vessel-col", "--moving-vessel-col", dest="faf_vessel_col", default="fixed_vessel_mask",
                   help="CSV column with the FAF vessel-mask path.")
    p.add_argument("--fov-scale", type=float, default=55.0 / 30.0,
                   help="FOV scale seed (default 55/30 ≈ 1.833).")
    p.add_argument("--size", type=int, default=224,
                   help="Working resolution, multiple of 14 (default 224).")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Skip CLAHE/blur preprocessing.")
    p.add_argument("--no-loftr", action="store_true",
                   help="Disable LoFTR feature matching (flow only).")
    p.add_argument("--scale-search", "--scale_search", dest="scale_search",
                   nargs="?", const=True, default=True, type=float,
                   help="Enable the narrow scale search around the seed, optionally with a half-width value.")
    p.add_argument("--no-images", action="store_true",
                   help="Only write results.csv; skip saving diagnostic images.")
    p.add_argument("--lowres-retry-size", type=int, default=336,
                   help="If a 224px run looks weak, retry this internal size and keep the better result.")
    p.add_argument("--lowres-retry-dice", type=float, default=0.25,
                   help="Trigger retry when initial dice_after is below this threshold.")
    p.add_argument("--lowres-retry-highres-size", type=int, default=1022,
                   help="Optional second-stage retry size for stubborn weak cases.")
    p.add_argument("--lowres-retry-highres-dice", type=float, default=0.12,
                   help="If result remains below this dice, run second-stage highres retry.")
    p.add_argument("--lowres-retry-margin", type=float, default=0.0,
                   help="Minimum dice improvement required to accept retry result.")
    p.add_argument("--no-lowres-retry", action="store_true",
                   help="Disable low-resolution retry fallback.")
    p.add_argument("--good-images-only", action="store_true",
                   help="Skip a pair unless both vessel masks are present and good, "
                        "both images are readable, and both optic discs are detected.")
    # ── flagging: post-registration only (matching never altered) ─────────────
    p.add_argument("--flag-min-anat", type=float, default=None,
                   help="Flag pairs (verdict 'low_quality') whose final anatomical "
                        "score is below this value, e.g. 0.55. Registers every pair "
                        "first, then flags only the low-scoring ones. Off by default.")
    p.add_argument("--flag-sparse", action="store_true",
                   help="Flag pairs (verdict 'sparse'/'unsegmentable') whose vessel "
                        "mask is thin/sparse OR dots/speckle. Off by default.")
    p.add_argument("--flag-unsegmentable", action="store_true",
                   help="Flag ONLY dots/speckle masks with no coherent vessel tree "
                        "(verdict 'unsegmentable'); decent thin ('sparse') masks are "
                        "kept. Use this to catch noise masks without flagging usable "
                        "thin ones. Off by default.")
    p.add_argument("--flag-noise-over", type=float, default=0.0,
                   help="Flag pairs (verdict 'noisy') when the FAF/IR image noise "
                        "estimate exceeds this value, e.g. 16 — catches junk/static "
                        "captures whose masks look thin. Noisy images with BOTH "
                        "masks well-segmented are kept. 0 = off (default).")
    p.add_argument("--flag-noise-even-if-good", action="store_true",
                   help="With --flag-noise-over, flag noisy images even when their "
                        "vessel masks are well-segmented.")
    # ── sparse/unsegmentable threshold tuning (override assess_vessel_mask) ──
    p.add_argument("--sparse-min-density", type=float, default=None,
                   help="Sparse threshold: mask vessel-pixel fraction below this is "
                        "'sparse' (default 0.008). LOWER it to flag only the barest "
                        "masks and keep decent-thin ones.")
    p.add_argument("--sparse-min-extent", type=float, default=None,
                   help="Sparse structure signal: largest-component bbox side / image "
                        "side below this counts as a noise signal (default 0.22).")
    p.add_argument("--sparse-min-skeleton", type=int, default=None,
                   help="Sparse structure signal: total skeleton pixels below this "
                        "counts as a noise signal (default 70).")
    p.add_argument("--sparse-max-fragmentation", type=float, default=None,
                   help="Sparse structure signal: components/pixels above this counts "
                        "as a noise signal (default 0.05).")
    # ── ablation mode ────────────────────────────────────────────────────────
    p.add_argument("--ablation", action="store_true",
                   help="Run the leave-one-out ablation study instead of a "
                        "normal batch (no diagnostic images are saved).")
    p.add_argument("--configs", nargs="+", default=None, metavar="NAME",
                   help="Ablation only: subset of configurations to run "
                        f"(default all). Choices: {[n for n, _, _ in ABLATIONS]}")
    p.add_argument("--only-row", type=int, nargs="+", default=None, metavar="N",
                   help="Run only specific 1-based CSV row numbers (e.g. --only-row 79).")
    p.add_argument("--cascade-aggressive-rows", type=int, nargs="+", default=None, metavar="N",
                   help="In batch runs, treat these 1-based CSV rows as aggressive for cascade rollback tuning (e.g. --cascade-aggressive-rows 79 49).")
    return p


def _flag_kwargs(args) -> dict:
    """Build the register() flag keyword arguments from CLI args."""
    # Only include thresholds the user actually overrode (else assess defaults).
    _sparse = {}
    if args.sparse_min_density is not None:
        _sparse["min_density"] = args.sparse_min_density
    if args.sparse_min_extent is not None:
        _sparse["min_largest_extent_frac"] = args.sparse_min_extent
    if args.sparse_min_skeleton is not None:
        _sparse["min_skeleton_pixels"] = args.sparse_min_skeleton
    if args.sparse_max_fragmentation is not None:
        _sparse["max_fragmentation"] = args.sparse_max_fragmentation

    return dict(
        flag_min_anat=args.flag_min_anat,
        flag_sparse=args.flag_sparse,
        flag_unsegmentable=args.flag_unsegmentable,
        sparse_thresholds=(_sparse or None),
        flag_noise_over=(args.flag_noise_over
                         if args.flag_noise_over and args.flag_noise_over > 0
                         else None),
        flag_noise_keep_if_segmentable=not args.flag_noise_even_if_good,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(sub: pd.DataFrame, stem: str, phase: str = "after"):
    """(mean, std, n) of a metric column, or None if unusable."""
    col = f"{stem}_{phase}"
    if col not in sub.columns:
        return None
    v = pd.to_numeric(sub[col], errors="coerce").dropna()
    if v.empty:
        return None
    return float(v.mean()), float(v.std(ddof=0)), int(len(v))


# ─────────────────────────────────────────────────────────────────────────────
# Normal-mode summary (before/after/delta, overall)
# ──────────────────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame):
    """Print mean ± std of before/after metrics, overall.

    Returns a tidy DataFrame (one row per group x metric) or None.
    """
    ok = df[df.get("status") == "ok"] if "status" in df.columns else df
    if ok.empty:
        print("\nNo successful pairs — nothing to summarize.")
        return None

    rows = []

    def _pick_metric_stem(sub: pd.DataFrame, preferred: str, fallback: str | None):
        if f"{preferred}_before" in sub.columns and f"{preferred}_after" in sub.columns:
            return preferred
        if fallback is not None and f"{fallback}_before" in sub.columns and f"{fallback}_after" in sub.columns:
            return fallback
        return None

    def _print_metric_block(sub: pd.DataFrame, group: str, metrics: list[tuple[str, str, str | None]],
                            prefix: str):
        print(f"\n--- {prefix} ---")
        print(f"  {'metric':<12} {'before':>22} {'after':>22} {'delta (a-b)':>22}")
        for shown, preferred, fallback in metrics:
            m = _pick_metric_stem(sub, preferred, fallback)
            if m is None:
                continue
            cb, ca = f"{m}_before", f"{m}_after"
            b = pd.to_numeric(sub[cb], errors="coerce")
            a = pd.to_numeric(sub[ca], errors="coerce")
            valid = b.notna() & a.notna()
            if not valid.any():
                continue
            b, a = b[valid], a[valid]
            d = a - b

            def ms(x):
                return f"{x.mean():10.4f} ± {x.std(ddof=0):8.4f}"

            print(f"  {shown:<12} {ms(b):>22} {ms(a):>22} {ms(d):>22}")
            rows.append(dict(
                group=group, category=prefix.lower(), metric=shown,
                metric_stem=m,
                n=int(valid.sum()),
                before_mean=round(float(b.mean()), 4),
                before_std=round(float(b.std(ddof=0)), 4),
                after_mean=round(float(a.mean()), 4),
                after_std=round(float(a.std(ddof=0)), 4),
                delta_mean=round(float(d.mean()), 4),
                delta_std=round(float(d.std(ddof=0)), 4),
            ))

    def _block(sub: pd.DataFrame, group: str):
        print(f"\n=== {group}  (n={len(sub)}) ===")
        # Preferred: explicit vessel/image metrics requested by user.
        printed_any = False
        if any(_pick_metric_stem(sub, pref, fb) is not None
               for _, pref, fb in SUMMARY_METRICS_VESSEL):
            _print_metric_block(sub, group, SUMMARY_METRICS_VESSEL, "Vessel")
            printed_any = True
        if any(_pick_metric_stem(sub, pref, fb) is not None
               for _, pref, fb in SUMMARY_METRICS_IMAGE):
            _print_metric_block(sub, group, SUMMARY_METRICS_IMAGE, "Image")
            printed_any = True

        # Backward-compat fallback for older result CSVs.
        if not printed_any:
            legacy = [(m, m, None) for m in SUMMARY_METRICS]
            _print_metric_block(sub, group, legacy, "Legacy")

    _block(ok, "ALL")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation-mode summary + table
# ─────────────────────────────────────────────────────────────────────────────

def summarize_ablation(df: pd.DataFrame, configs) -> pd.DataFrame:
    """Tidy per-(config x metric) summary of the *after* metrics."""
    rows = []
    ok = df[df["status"] == "ok"]
    for name, label, _ in configs:
        sub = ok[ok["config"] == name]
        for stem, header, higher in ABL_METRICS:
            st = _stats(sub, stem)
            if st is None:
                continue
            mean, std, n = st
            rows.append(dict(config=name, label=label, metric=stem,
                             n=n, mean=round(mean, 4), std=round(std, 4),
                             higher_is_better=higher))
    return pd.DataFrame(rows)


def print_ablation_table(df: pd.DataFrame, configs) -> str:
    """Print the console table and return the markdown version."""
    ok = df[df["status"] == "ok"]

    cell = {}
    for name, _, _ in configs:
        sub = ok[ok["config"] == name]
        for stem, _, _ in ABL_METRICS:
            cell[(name, stem)] = _stats(sub, stem)

    # Best config per metric (by mean).
    best = {}
    for stem, _, higher in ABL_METRICS:
        vals = [(n, cell[(n, stem)][0]) for n, _, _ in configs
                if cell.get((n, stem)) is not None]
        if vals:
            best[stem] = (max if higher else min)(vals, key=lambda x: x[1])[0]

    def fmt(name, stem, bold=False):
        st = cell.get((name, stem))
        if st is None:
            return "—"
        s = f"{st[0]:.4f} ± {st[1]:.4f}"
        return f"**{s}**" if bold else s

    def ticks(overrides):
        return ["✓" if overrides.get(kw, True) else "✗"
                for _, kw in COMPONENTS]

    # ── console ──
    comp_hdr = " ".join(f"{h:^12}" for h, _ in COMPONENTS)
    met_hdr  = " ".join(f"{h:^19}" for _, h, _ in ABL_METRICS)
    print(f"\n{'Experiment':<24} {comp_hdr} {met_hdr}")
    for name, label, ov in configs:
        comp = " ".join(f"{t:^12}" for t in ticks(ov))
        mets = " ".join(f"{fmt(name, stem):^19}" for stem, _, _ in ABL_METRICS)
        print(f"{label:<24} {comp} {mets}")

    # ── markdown ──
    hdr = (["Experiment"] + [h for h, _ in COMPONENTS]
           + [h for _, h, _ in ABL_METRICS])
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for name, label, ov in configs:
        cells = ([label] + ticks(ov)
                 + [fmt(name, stem, bold=(best.get(stem) == name))
                    for stem, _, _ in ABL_METRICS])
        lines.append("| " + " | ".join(cells) + " |")
    n_pairs = ok.groupby("config").size().max() if not ok.empty else 0
    lines.append(f"\nValues are mean ± std over {int(n_pairs)} pairs "
                 f"(after registration). Best per column in bold. "
                 f"↑ higher is better, ↓ lower is better.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Row iteration shared by both modes
# ─────────────────────────────────────────────────────────────────────────────

def _session_tag(path_str: str) -> str:
    """Return the nearest one or two parent-folder names (e.g. patient/session
    IDs) for a path, so output filenames stay unique even when the leaf image
    filename (e.g. ``AF_B-0_0.png``) repeats across different patients or
    visits. Falls back to an empty string if the path is too shallow."""
    parts = Path(str(path_str)).parts
    parents = [p for p in parts[:-1] if p not in ("", "/")]
    tag_parts = parents[-2:] if len(parents) >= 2 else parents[-1:]
    return "-".join(tag_parts)


def _iter_pairs(df, args):
    for n, (_, row) in enumerate(df.iterrows(), 1):
        ir_path = row[args.ir_col]
        faf_path = row[args.faf_col]
        # Vessel columns are optional. Missing/blank values are passed as
        # ``None`` and the pipeline automatically uses image-based matching.
        ir_vessel_path = row.get(args.ir_vessel_col, None)
        faf_vessel_path = row.get(args.faf_vessel_col, None)
        faf_tag = _session_tag(faf_path)
        ir_tag = _session_tag(ir_path)
        faf_label = f"{faf_tag}_{Path(str(faf_path)).stem}" if faf_tag else Path(str(faf_path)).stem
        ir_label = f"{ir_tag}_{Path(str(ir_path)).stem}" if ir_tag else Path(str(ir_path)).stem
        name = f"{n:04d}_{faf_label}_to_{ir_label}"
        yield n, name, ir_path, faf_path, ir_vessel_path, faf_vessel_path


def _good_image_only_gate(ir_path, faf_path, ir_vessel_path, faf_vessel_path,
                          size: int):
    """Return ``(allowed, reason, quality)`` for strict batch screening.

    This intentionally does not derive pseudo-vessel masks or attempt image
    fallback.  It is for batches where only clearly usable vessel-and-disc
    pairs should be registered.
    """
    paths = {
        "ir_vessel_mask_missing": ir_vessel_path,
        "faf_vessel_mask_missing": faf_vessel_path,
    }
    missing = [reason for reason, path in paths.items()
               if path is None or not str(path).strip() or not Path(str(path)).exists()]
    if missing:
        return False, ";".join(missing), "bad"

    try:
        fi = load_image(ir_path, size, size)
        mi = load_image(faf_path, size, size)
        fv = load_vessel(ir_vessel_path, size, size)
        mv = load_vessel(faf_vessel_path, size, size)
    except Exception as exc:
        return False, f"input_unreadable:{exc}", "bad"

    quality, reasons = classify_pair_quality(fv, mv, fi=fi, mi=mi)
    if quality != "good":
        return False, ";".join(reasons) or "vessel_masks_not_good", quality

    unreadable, readability_reason = classify_pair_readability(fi, mi)
    if unreadable:
        return False, f"unreadable:{readability_reason}", quality

    return True, "", quality


def _should_retry_lowres(result, args) -> bool:
    if args.no_lowres_retry:
        return False
    if int(args.size) > 224:
        return False
    if int(args.lowres_retry_size) <= int(args.size):
        return False
    dice = result.metrics.get("dice_after") if result.metrics else None
    low_dice = (dice is not None) and (float(dice) < float(args.lowres_retry_dice))
    weak_label = str(result.label).startswith("scale_best")
    return bool(low_dice or weak_label)


def _pick_better_result(base, retry, margin: float):
    d0 = base.metrics.get("dice_after") if base.metrics else None
    d1 = retry.metrics.get("dice_after") if retry.metrics else None
    if d1 is None:
        return base
    if d0 is None:
        return retry
    if float(d1) > float(d0) + float(margin):
        return retry
    return base


def _need_highres_retry(result, args) -> bool:
    if args.no_lowres_retry:
        return False
    d = result.metrics.get("dice_after") if result.metrics else None
    if d is None:
        return False
    return float(d) < float(args.lowres_retry_highres_dice)


def _round14(x: float) -> int:
    """Round to the nearest positive multiple of 14 (register() requires it)."""
    return max(14, int(round(x / 14.0)) * 14)


def _resolution_fallback_ladder(base_size: int, cap: int, step: int) -> list:
    """Build an increasing ladder of working resolutions (multiples of 14)
    from just above ``base_size`` up to ``cap`` inclusive.
    """
    step = max(_round14(step), 14)
    sizes = []
    s = _round14(base_size + step)
    while s < cap:
        sizes.append(s)
        s += step
    if cap > base_size and (not sizes or sizes[-1] != cap):
        sizes.append(int(cap))
    return sizes


def _register_with_failure_upgrade(register_kwargs: dict, args):
    """Call register() at the baseline size; if that raises a hard failure,
    retry at progressively higher resolutions until one succeeds or the ladder
    reaches the cap. The baseline size is never changed."""
    base_size = int(register_kwargs.get("size"))
    try:
        return register(**register_kwargs)
    except Exception as e:
        if args.no_lowres_retry:
            raise
        cap = int(args.lowres_retry_highres_size)
        step = int(args.lowres_retry_size) - base_size
        candidate_sizes = _resolution_fallback_ladder(base_size, cap, step)
        last_err = e
        for sz in candidate_sizes:
            print(f"  [FailureUpgrade] {base_size}px failed ({last_err}); "
                  f"retrying at {sz}px")
            kwargs = dict(register_kwargs)
            kwargs["size"] = sz
            try:
                result = register(**kwargs)
            except Exception as e2:
                last_err = e2
                continue
            print(f"  [FailureUpgrade] succeeded at {sz}px")
            result.label = f"{result.label}|failupgrade{sz}"
            return result
        raise last_err


# ─────────────────────────────────────────────────────────────────────────────
# Ablation driver  (flagging is intentionally NOT applied in ablation)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(df: pd.DataFrame, out_dir: Path, args) -> int:
    configs = ABLATIONS
    if args.configs:
        known = {n for n, _, _ in ABLATIONS}
        bad = [c for c in args.configs if c not in known]
        if bad:
            print(f"error: unknown config(s) {bad}; "
                  f"choices: {sorted(known)}", file=sys.stderr)
            return 2
        configs = [c for c in ABLATIONS if c[0] in args.configs]

    n_runs = len(df) * len(configs)
    print(f"ablation: {len(df)} pair(s) x {len(configs)} config(s) "
          f"= {n_runs} registrations  ->  {out_dir.resolve()}\n")

    results_csv = out_dir / "ablation_results.csv"
    records = []
    for n, pair, ir_path, faf_path, ir_v, faf_v in _iter_pairs(df, args):
        # honor --only-row if provided
        if args.only_row and (n not in args.only_row):
            continue
        print(f"[{n}/{len(df)}] {Path(str(faf_path)).name} -> "
              f"{Path(str(ir_path)).name}")
        for name, label, overrides in configs:
            print(f"  -- config: {name}")
            try:
                utils.CURRENT_ROW = n
                result = _register_with_failure_upgrade(dict(
                    ir_image=ir_path, faf_image=faf_path,
                    ir_vessel=ir_v, faf_vessel=faf_v,
                    fov_scale=args.fov_scale, size=args.size,
                    **overrides,
                ), args)
                if _should_retry_lowres(result, args):
                    retry = register(
                        ir_image=ir_path, faf_image=faf_path,
                        ir_vessel=ir_v, faf_vessel=faf_v,
                        fov_scale=args.fov_scale, size=args.lowres_retry_size,
                        **overrides,
                    )
                    chosen = _pick_better_result(result, retry, args.lowres_retry_margin)
                    if chosen is retry:
                        retry.label = f"{retry.label}|retry{args.lowres_retry_size}"
                        result = retry
                if (_need_highres_retry(result, args)
                        and int(args.lowres_retry_highres_size) > int(args.lowres_retry_size)):
                    retry_hi = register(
                        ir_image=ir_path, faf_image=faf_path,
                        ir_vessel=ir_v, faf_vessel=faf_v,
                        fov_scale=args.fov_scale, size=args.lowres_retry_highres_size,
                        **overrides,
                    )
                    chosen = _pick_better_result(result, retry_hi, args.lowres_retry_margin)
                    if chosen is retry_hi:
                        retry_hi.label = f"{retry_hi.label}|retry{args.lowres_retry_highres_size}"
                        result = retry_hi
            except Exception as e:  # keep going on bad rows
                print(f"     ERROR: {e}")
                records.append(dict(pair=pair, config=name, ir=ir_path,
                                    faf=faf_path, status="error", error=str(e)))
                continue
            finally:
                utils.CURRENT_ROW = None
            records.append(dict(
                pair=pair, config=name, ir=ir_path, faf=faf_path,
                status="ok", label=result.label, scale=result.scale,
                **result.metrics,
            ))
        # Crash-safe: rewrite the results CSV after every pair.
        pd.DataFrame(records).to_csv(results_csv, index=False)

    df_out = pd.DataFrame(records)
    df_out.to_csv(results_csv, index=False)

    summary = summarize_ablation(df_out, configs)
    summary_csv = out_dir / "ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    md = print_ablation_table(df_out, configs)
    table_md = out_dir / "ablation_table.md"
    table_md.write_text(md, encoding="utf-8")

    ok = int((df_out["status"] == "ok").sum())
    print(f"\nDone. {ok}/{len(records)} runs ok.")
    print(f"Per-run metrics -> {results_csv.resolve()}")
    print(f"Summary         -> {summary_csv.resolve()}")
    print(f"Markdown table  -> {table_md.resolve()}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Normal batch driver
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(df: pd.DataFrame, out_dir: Path, args) -> int:
    print(f"fafir-register: {len(df)} pair(s)  ->  {out_dir.resolve()}\n")

    # Apply the optional scale-search half-width once, before the loop.
    if isinstance(args.scale_search, (int, float)) and not isinstance(args.scale_search, bool):
        utils.SCALE_SEARCH_HALFWIDTH = float(args.scale_search)

    fk = _flag_kwargs(args)   # register() flag arguments, shared by all calls

    records = []
    for n, name, ir_path, faf_path, ir_v, faf_v in _iter_pairs(df, args):
        # Optional fast-path: only run explicitly-requested rows.
        if args.only_row and (n not in args.only_row):
            continue
        print(f"[{n}/{len(df)}] {Path(str(faf_path)).name} -> {Path(str(ir_path)).name}")
        if args.good_images_only:
            allowed, reason, quality = _good_image_only_gate(
                ir_path, faf_path, ir_v, faf_v, args.size)
            if not allowed:
                print(f"  SKIPPED: {reason}")
                records.append(dict(
                    name=name, ir=ir_path, faf=faf_path,
                    status="skipped", quality=quality, skip_reason=reason,
                ))
                continue
        try:
            utils.CURRENT_ROW = n
            result = _register_with_failure_upgrade(dict(
                ir_image=ir_path, faf_image=faf_path,
                ir_vessel=ir_v, faf_vessel=faf_v,
                fov_scale=args.fov_scale, size=args.size,
                preprocess=not args.no_preprocess,
                use_loftr=not args.no_loftr,
                scale_search=bool(args.scale_search),
                **fk,
            ), args)
            # A flagged pair must NOT be retried — retries would re-run
            # registration and overwrite (un-flag) the result.
            _flagged = getattr(result, "status", "ok") == "flagged"
            if not _flagged and _should_retry_lowres(result, args):
                retry = register(
                    ir_image=ir_path, faf_image=faf_path,
                    ir_vessel=ir_v, faf_vessel=faf_v,
                    fov_scale=args.fov_scale, size=args.lowres_retry_size,
                    preprocess=not args.no_preprocess,
                    use_loftr=not args.no_loftr,
                    scale_search=bool(args.scale_search),
                    **fk,
                )
                chosen = _pick_better_result(result, retry, args.lowres_retry_margin)
                if chosen is retry:
                    retry.label = f"{retry.label}|retry{args.lowres_retry_size}"
                    result = retry
            if (not _flagged and _need_highres_retry(result, args)
                    and int(args.lowres_retry_highres_size) > int(args.lowres_retry_size)):
                retry_hi = register(
                    ir_image=ir_path, faf_image=faf_path,
                    ir_vessel=ir_v, faf_vessel=faf_v,
                    fov_scale=args.fov_scale, size=args.lowres_retry_highres_size,
                    preprocess=not args.no_preprocess,
                    use_loftr=not args.no_loftr,
                    scale_search=bool(args.scale_search),
                    **fk,
                )
                chosen = _pick_better_result(result, retry_hi, args.lowres_retry_margin)
                if chosen is retry_hi:
                    retry_hi.label = f"{retry_hi.label}|retry{args.lowres_retry_highres_size}"
                    result = retry_hi
        except Exception as e:  # keep going on bad rows
            print(f"  ERROR: {e}")
            records.append(dict(name=name, ir=ir_path, faf=faf_path,
                                status="error", error=str(e)))
            continue
        finally:
            utils.CURRENT_ROW = None

        # Flagged pair: save the flagged grid to flagged/ and record it.
        if getattr(result, "status", "ok") == "flagged":
            print(f"  FLAGGED ({result.flag_verdict}): {result.flag_reason}")
            if not args.no_images:
                result.save(out_dir / "flagged", name=name)
            records.append(dict(
                name=name, ir=ir_path, faf=faf_path, status="flagged",
                quality=result.flag_verdict, skip_reason=result.flag_reason))
            continue

        if not args.no_images:
            result.save(out_dir, name=name)

        records.append(dict(
            name=name, ir=ir_path, faf=faf_path, status="ok",
            label=result.label, scale=result.scale,
            **result.metrics,
        ))

    results_csv = out_dir / "results.csv"
    df_out = pd.DataFrame(records)
    df_out.to_csv(results_csv, index=False)

    summary = summarize(df_out)
    if summary is not None:
        summary_csv = out_dir / "summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"\nSummary -> {summary_csv.resolve()}")

    ok = sum(1 for r in records if r.get("status") == "ok")
    flagged = sum(1 for r in records if r.get("status") == "flagged")
    print(f"Done. {ok}/{len(records)} ok.  Metrics -> {results_csv.resolve()}")
    if flagged:
        print(f"Flagged: {flagged}/{len(records)}  ->  {(out_dir / 'flagged').resolve()}")
    return 0


def main(argv=None) -> int:
    _seed_everything(0)   # reproducible LoFTR / RANSAC / scoring
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Vessel masks are optional.
    required = [args.ir_col, args.faf_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"error: CSV missing columns: {missing}\n"
              f"       available: {list(df.columns)}", file=sys.stderr)
        return 2

    # If the user requested `--only-row`, expose it to the registration logic
    # as a set of rows for aggressive cascade rollback tuning.
    # Allow explicit per-row aggressive cascade tuning for batch runs.
    if getattr(args, "cascade_aggressive_rows", None):
        utils.CASCADE_AGGRESSIVE_ROWS = set(args.cascade_aggressive_rows)
    # Backwards-compatible: if the user only supplied `--only-row` (single-row
    # debug runs), expose those rows as aggressive too.
    if args.only_row and not getattr(args, "cascade_aggressive_rows", None):
        utils.CASCADE_AGGRESSIVE_ROWS = set(args.only_row)

    if args.ablation:
        return run_ablation(df, out_dir, args)
    return run_batch(df, out_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())