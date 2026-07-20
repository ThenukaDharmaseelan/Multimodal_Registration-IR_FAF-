"""fafirs_registration.io — image and vessel-mask loading.

Thin wrappers over ``cv2.imread`` that load, colour-convert and resize to the
working resolution. Images are returned as (H, W, 3) uint8 RGB; vessel masks as
(H, W) float32 with values in {0, 1}. Both raise ``FileNotFoundError`` if the
path cannot be read.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path, h, w):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (w, h),
                      interpolation=cv2.INTER_LINEAR)


def load_vessel(path, h, w):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return (cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127
            ).astype(np.float32)