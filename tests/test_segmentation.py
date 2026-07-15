import unittest

import numpy as np

from sam3_service.inference import (
    clip_mask_to_padded_box,
    select_box_mask,
    xyxy_to_normalized_cxcywh,
)
from segmentation.masks import combine_masks, make_mask_overlay, scale_box_xyxy


class Sam3BoxTests(unittest.TestCase):
    def test_converts_xyxy_to_normalized_cxcywh(self):
        converted = xyxy_to_normalized_cxcywh([20, 10, 60, 30], width=100, height=50)
        np.testing.assert_allclose(converted, [0.4, 0.4, 0.4, 0.4])

    def test_selects_candidate_inside_detector_box(self):
        candidates = np.zeros((2, 10, 10), dtype=bool)
        candidates[0, 0:2, 0:2] = True
        candidates[1, 4:8, 4:8] = True
        mask, score = select_box_mask(candidates, np.array([0.9, 0.7]), [4, 4, 8, 8])
        np.testing.assert_array_equal(mask, candidates[1])
        self.assertAlmostEqual(score, 0.7)

    def test_clips_mask_to_padded_box(self):
        mask = np.ones((10, 10), dtype=bool)
        clipped = clip_mask_to_padded_box(mask, [3, 3, 7, 7], padding=0.0)
        self.assertEqual(int(clipped.sum()), 16)


class OverlayTests(unittest.TestCase):
    def test_scales_square_detector_box_to_original_image(self):
        box = scale_box_xyxy([320, 160, 640, 640], (640, 640), (1920, 1080))
        np.testing.assert_allclose(box, [960, 270, 1920, 1080])

    def test_combines_masks_and_renders_overlay(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        masks = np.zeros((1, 8, 8), dtype=bool)
        masks[0, 2:6, 2:6] = True
        combined = combine_masks(masks, (8, 8))
        overlay = make_mask_overlay(image, masks, [[2, 2, 6, 6]], ["Crack 0.80"])
        self.assertEqual(combined.dtype, np.uint8)
        self.assertEqual(int(combined.sum()), 16 * 255)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)


if __name__ == "__main__":
    unittest.main()
