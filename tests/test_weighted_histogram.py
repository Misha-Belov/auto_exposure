import unittest

import numpy as np

from src.raw_proc_estim import (
    calculate_weighted_histogram,
    create_center_weights,
    estimate_clipping,
    estimate_exposure,
    find_unclipped_median,
)


class WeightedHistogramTests(unittest.TestCase):
    def test_weights_are_symmetric_and_decrease_to_edges(self):
        weights = create_center_weights((5, 7))
        self.assertEqual(weights[2, 3], 1)
        np.testing.assert_allclose(weights[0], 0.1)
        np.testing.assert_allclose(weights[:, 0], 0.1)
        np.testing.assert_allclose(weights, weights[::-1, ::-1])
        self.assertGreater(weights[2, 3], weights[1, 3])
        self.assertGreater(weights[1, 3], weights[0, 3])

    def test_uniform_weights_restore_pixel_counts(self):
        raw = np.array([[0, 20, 30], [30, 100, 0]], dtype=np.uint16)
        weights = create_center_weights(raw.shape, edge_weight=1)
        actual = calculate_weighted_histogram(raw, 100, weights)
        expected = np.bincount(raw.ravel(), minlength=101)
        np.testing.assert_array_equal(actual, expected)

    def test_central_region_can_outweigh_more_peripheral_pixels(self):
        raw = np.full((5, 5), 20, dtype=np.uint16)
        raw[1:4, 1:4] = 80
        weights = create_center_weights(raw.shape)
        weighted = calculate_weighted_histogram(raw, 100, weights)
        uniform = np.bincount(raw.ravel(), minlength=101)
        self.assertEqual(find_unclipped_median(uniform, 100), 20)
        self.assertEqual(find_unclipped_median(weighted, 100), 80)
        self.assertLess(estimate_exposure(uniform, 100), 0)
        self.assertGreater(estimate_exposure(weighted, 100), 0)
        self.assertAlmostEqual(weighted.sum(), weights.sum())

    def test_central_clipping_has_more_influence(self):
        weights = create_center_weights((5, 5))
        for clipped_value, sign in ((0, -1), (100, 1)):
            scores = []
            for position in ((0, 0), (2, 2)):
                raw = np.full((5, 5), 50, dtype=np.uint16)
                raw[position] = clipped_value
                hist = calculate_weighted_histogram(raw, 100, weights)
                score = estimate_clipping(hist, 100)
                self.assertGreater(sign * score, 0)
                # Оценка не зависит от общего масштаба весов.
                self.assertAlmostEqual(score, estimate_clipping(hist / 100, 100))
                scores.append(abs(score))
            self.assertGreater(scores[1], scores[0])

    def test_fractional_clipping_extremes(self):
        for value, expected in ((0, -np.inf), (100, np.inf)):
            raw = np.full((2, 2), value, dtype=np.uint16)
            hist = calculate_weighted_histogram(raw, 100, np.full((2, 2), 0.1))
            self.assertEqual(estimate_clipping(hist, 100), expected)

    def test_small_frames_and_input_preservation(self):
        for shape in ((1, 1), (1, 5), (4, 1), (2, 2)):
            raw = np.full(shape, 50, dtype=np.uint16)
            weights = create_center_weights(shape)
            original_weights = weights.copy()
            hist = calculate_weighted_histogram(raw, 100, weights)
            self.assertEqual(weights.shape, raw.shape)
            self.assertGreater(hist.sum(), 0)
            self.assertEqual(estimate_exposure(hist, 100), 0)
            np.testing.assert_array_equal(raw, np.full(shape, 50))
            np.testing.assert_array_equal(weights, original_weights)


if __name__ == "__main__":
    unittest.main()
