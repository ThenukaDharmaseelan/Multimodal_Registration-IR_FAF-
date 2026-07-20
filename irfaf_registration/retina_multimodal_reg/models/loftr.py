"""fafir_registration.models.loftr — LoFTR matcher state and loader.

The LoFTR model (from ``kornia``) is optional. If kornia is not installed the
package still runs, falling back to optical-flow-only registration. Model state
is held here as module attributes and read by :mod:`fafir_registration.matching`
via ``loftr.LOFTR_AVAILABLE`` / ``loftr.loftr_matcher``.
"""

from __future__ import annotations

from .. import utils

# Module-level model state (mutated by init_loftr).
LOFTR_AVAILABLE = False
loftr_matcher = None


def init_loftr():
    """Load the pretrained LoFTR matcher onto the active device.

    Safe to call repeatedly. On failure (kornia missing, download blocked) it
    prints a notice and leaves ``LOFTR_AVAILABLE`` False so callers fall back to
    flow-only matching."""
    global LOFTR_AVAILABLE, loftr_matcher
    try:
        from kornia.feature import LoFTR
        matcher = LoFTR(pretrained="outdoor")
        matcher = matcher.to(utils.DEVICE).eval()
        loftr_matcher = matcher
        LOFTR_AVAILABLE = True
        print("  LoFTR loaded successfully")
    except Exception as e:  # pragma: no cover - depends on environment
        print(f"  LoFTR not available ({e}) — using flow only")
        LOFTR_AVAILABLE = False