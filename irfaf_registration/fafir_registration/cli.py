"""Compatibility entry point for ``python -m fafir_registration.cli``."""

from __future__ import annotations

from retina_multimodal_reg.cli import *  # noqa: F401,F403

if __name__ == "__main__":
    from retina_multimodal_reg.cli import main

    raise SystemExit(main())