"""Legacy compatibility package for the old ``fafir_registration`` name.

The implementation now lives in :mod:`retina_multimodal_reg`, but some tests
and scripts still import the historical package name. This shim re-exports the
public API and registers the commonly used submodules under the legacy import
path.
"""

from __future__ import annotations

import importlib
import sys

_MODULES = [
    "cli",
    "flow",
    "io",
    "matching",
    "metrics",
    "pipeline",
    "preprocessing",
    "registration",
    "utils",
    "visualization",
]

_impl = importlib.import_module("retina_multimodal_reg")

register = _impl.register
RegistrationResult = _impl.RegistrationResult
metrics = _impl.metrics
visualization = _impl.visualization
__version__ = _impl.__version__

__all__ = list(getattr(_impl, "__all__", ()))

for module_name in _MODULES:
    module = importlib.import_module(f"retina_multimodal_reg.{module_name}")
    sys.modules[f"{__name__}.{module_name}"] = module
    globals()[module_name] = module
