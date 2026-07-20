"""fafir_registration — FAF↔IR retinal image registration.

A modular, stable API over a validated FOV-seed + LoFTR + optical-flow
registration pipeline. The package is split into focused modules:

    utils           configuration, geometry, affine sanity
    io              image / vessel-mask loading
    preprocessing   CLAHE, vessel closing, density heuristics
    models.loftr    LoFTR matcher state + loader
    metrics         Dice/NCC/HD95/ASD/Wasserstein + scoring objectives
    matching        LoFTR -> affine
    flow            optical-flow refinement + cascade
    registration    candidate generation + selection
    visualization   diagnostic images + grids
    pipeline        the public ``register`` entry point

Basic use::

    from fafir_registration import register

    result = register(
        ir_image="ir.png",
        faf_image="faf.png",
        ir_vessel="ir_vessel.png",
        faf_vessel="faf_vessel.png",
    )
    result.transform          # 2x3 affine (moving -> fixed)
    result.registered_image   # warped FAF
    result.metrics            # before/after quality metrics

Batch use from the shell::

    fafir-register pairs.csv output/
"""

from . import metrics, visualization
from .pipeline import RegistrationResult, register

__version__ = "0.1.0"

__all__ = [
    "register",
    "RegistrationResult",
    "metrics",
    "visualization",
    "__version__",
]