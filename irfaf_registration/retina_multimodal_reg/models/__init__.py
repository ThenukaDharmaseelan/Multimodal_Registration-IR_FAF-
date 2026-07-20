"""fafir_registration.models — learned models used by the pipeline.

Currently this houses the LoFTR feature matcher. Model state (the loaded
matcher and an availability flag) lives in :mod:`fafir_registration.models.loftr`
and is initialized lazily via :func:`fafir_registration.models.loftr.init_loftr`.
"""

from . import loftr

__all__ = ["loftr"]