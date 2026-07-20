import inspect
import importlib
import unittest
from unittest.mock import patch

from fafir_registration import cli, utils
from fafir_registration.pipeline import register as pipeline_register
from fafir_registration.preprocessing import classify_pair_quality
from fafir_registration import matching, registration
from fafir_registration.preprocessing import (
    estimate_optic_disc_center,
    estimate_vessel_mask,
)
from fafir_registration.pipeline import _as_vessel, _vessel_mask_missing


def test_scale_search_window_uses_seed_centered_halfwidth():
    seed = 0.545
    original_lo, original_hi = utils.SCALE_SEARCH_LO, utils.SCALE_SEARCH_HI
    utils.SCALE_SEARCH_LO, utils.SCALE_SEARCH_HI = None, None
    try:
        lo, hi = utils.scale_search_window(seed)
        assert lo == seed - utils.SCALE_SEARCH_HALFWIDTH
        assert hi == seed + utils.SCALE_SEARCH_HALFWIDTH
    finally:
        utils.SCALE_SEARCH_LO, utils.SCALE_SEARCH_HI = original_lo, original_hi


def test_scale_search_window_honors_explicit_bounds():
    seed = 0.545
    original_lo, original_hi = utils.SCALE_SEARCH_LO, utils.SCALE_SEARCH_HI
    utils.SCALE_SEARCH_LO, utils.SCALE_SEARCH_HI = 0.45, 0.65
    try:
        lo, hi = utils.scale_search_window(seed)
        assert (lo, hi) == (0.45, 0.65)
    finally:
        utils.SCALE_SEARCH_LO, utils.SCALE_SEARCH_HI = original_lo, original_hi


def test_cli_defaults_to_scale_search_enabled():
    parser = cli.build_parser()
    args = parser.parse_args(["pairs.csv", "out"])
    assert args.scale_search is True


def test_cli_good_images_only_is_opt_in():
    parser = cli.build_parser()
    assert parser.parse_args(["pairs.csv", "out"]).good_images_only is False
    assert parser.parse_args(["pairs.csv", "out", "--good-images-only"]).good_images_only is True


def test_pipeline_register_defaults_to_scale_search_enabled():
    assert inspect.signature(pipeline_register).parameters["scale_search"].default is True


def test_sparse_visible_vessels_are_not_labelled_no_vessels():
    import numpy as np

    fv = np.zeros((224, 224), dtype=np.float32)
    mv = np.zeros((224, 224), dtype=np.float32)
    fv[80:84, 80:84] = 1.0
    mv[80:84, 80:84] = 1.0

    quality, reasons = classify_pair_quality(fv, mv)

    assert quality == "moderate"
    assert reasons[0].startswith("sparse_vessels")


def test_effectively_empty_mask_is_labelled_no_vessels():
    import numpy as np

    fv = np.zeros((224, 224), dtype=np.float32)
    mv = np.zeros((224, 224), dtype=np.float32)
    fv[80:84, 80:84] = 1.0

    quality, reasons = classify_pair_quality(fv, mv)

    assert quality == "bad"
    assert reasons[0].startswith("no_vessels")


def test_image_loftr_input_masks_black_border_and_retains_edges():
    import numpy as np

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[8:56, 8:56] = 80
    image[24:40, 24:40] = 180

    representation = matching._img_to_loftr_input(image)

    assert representation.dtype == np.float32
    assert np.all(representation[:4] == 0.0)
    assert representation[24:40, 24:40].max() > 0.0


def test_image_alignment_score_prioritizes_nmi(monkeypatch):
    import numpy as np

    fake_metrics = {"nmi": 1.2, "ncc_img": 0.4}
    monkeypatch.setattr(registration, "image_intensity_metrics",
                        lambda *args, **kwargs: fake_metrics)
    monkeypatch.setattr(registration.utils, "warp_img",
                        lambda image, M, H, W: image)
    monkeypatch.setattr(registration.utils, "get_fov_mask",
                        lambda image: np.ones(image.shape[:2], dtype=np.float32))

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    score = registration._image_alignment_score(
        image, image, np.eye(2, 3, dtype=np.float32), 8, 8)

    assert score == 1.22


def test_estimate_optic_disc_center_finds_compact_bright_region():
    import cv2
    import numpy as np

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[8:120, 8:120] = 80
    cv2.circle(image, (44, 70), 9, (220, 220, 220), thickness=-1)

    center, confidence = estimate_optic_disc_center(image)

    assert center is not None
    assert np.allclose(center, (44, 70), atol=2.0)
    assert confidence > 0.0


def test_disc_anchor_aligns_detected_centers():
    import cv2
    import numpy as np

    fixed = np.zeros((128, 128, 3), dtype=np.uint8)
    moving = np.zeros((128, 128, 3), dtype=np.uint8)
    fixed[8:120, 8:120] = 80
    moving[8:120, 8:120] = 80
    cv2.circle(fixed, (44, 70), 9, (220, 220, 220), thickness=-1)
    cv2.circle(moving, (58, 52), 9, (220, 220, 220), thickness=-1)
    identity = np.eye(2, 3, dtype=np.float32)

    transform, meta = registration._disc_anchor_candidate(
        fixed, moving, identity, 128, 128)

    assert transform is not None
    assert meta is not None
    mapped = transform[:, :2] @ np.array(meta["moving_center"]) + transform[:, 2]
    assert np.allclose(mapped, meta["fixed_center"], atol=1e-4)


def test_image_fallback_threshold_marks_weak_vessel_candidate():
    assert utils.IMAGE_FALLBACK_MIN_ANATOMICAL == 0.45
    assert 0.30 < utils.IMAGE_FALLBACK_MIN_ANATOMICAL
    assert 0.60 > utils.IMAGE_FALLBACK_MIN_ANATOMICAL


def test_disc_marker_applies_affine_transform():
    import numpy as np
    from fafir_registration.visualization import _transform_point

    transform = np.array([[2.0, 0.0, -4.0], [0.0, 2.0, 6.0]], dtype=np.float32)
    assert _transform_point(transform, (10.0, 8.0)) == (16.0, 22.0)


def test_missing_vessel_mask_uses_empty_image_fallback_mask():
    import numpy as np

    assert _vessel_mask_missing(None)
    assert _vessel_mask_missing(float("nan"))
    mask = _as_vessel(None, 16, 16)
    assert mask.shape == (16, 16)
    assert mask.dtype == np.float32
    assert mask.sum() == 0.0


def test_cli_iter_pairs_allows_csv_without_vessel_columns():
    import pandas as pd

    args = cli.build_parser().parse_args(["pairs.csv", "out"])
    frame = pd.DataFrame({"moving": ["ir.png"], "fixed": ["faf.png"]})
    pair = next(cli._iter_pairs(frame, args))

    assert pair[4] is None
    assert pair[5] is None


def test_image_derived_vessel_mask_recovers_dark_vessel_like_structure():
    import cv2
    import numpy as np

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[8:120, 8:120] = 180
    cv2.line(image, (20, 30), (105, 92), (20, 20, 20), thickness=3)

    mask = estimate_vessel_mask(image)

    assert mask.sum() >= 32
    assert mask[61, 62] == 1.0


class TestAsdSelection(unittest.TestCase):
    def test_asd_selection_uses_anatomical_tiebreak(self):
        import numpy as np
        from fafir_registration import registration

        M1 = np.eye(2, 3, dtype=np.float32)
        M2 = np.array([[0.9, 0.0, 1.0], [0.0, 0.9, 2.0]], dtype=np.float32)
        candidates = [
            ("scale_best", M1, 0.600, 0.600),
            ("flow->affine", M2, 0.610, 0.610),
        ]

        def fake_asd_reward(M, fv, mv, params=None):
            return 0.952 if np.allclose(M, M1) else 0.950

        with patch.object(registration, "asd_reward", side_effect=fake_asd_reward):
            winner, reason, score = registration._final_select(candidates, None, None, None)

        self.assertEqual(winner[0], "flow->affine")
        self.assertEqual(reason, "asd+anatomical")
        self.assertEqual(score, 0.950)


def test_asd_selection_uses_anatomical_tiebreak(monkeypatch):
    import numpy as np
    from fafir_registration import registration

    M1 = np.eye(2, 3, dtype=np.float32)
    M2 = np.array([[0.9, 0.0, 1.0], [0.0, 0.9, 2.0]], dtype=np.float32)
    candidates = [
        ("scale_best", M1, 0.600, 0.600),
        ("flow->affine", M2, 0.610, 0.610),
    ]

    def fake_asd_reward(M, fv, mv, params=None):
        return 0.952 if np.allclose(M, M1) else 0.950

    monkeypatch.setattr(registration, "asd_reward", fake_asd_reward)
    winner, reason, score = registration._final_select(candidates, None, None, None)

    assert winner[0] == "flow->affine"
    assert reason == "asd+anatomical"
    assert score == 0.950
