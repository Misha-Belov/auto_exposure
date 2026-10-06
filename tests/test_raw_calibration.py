import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.raw_calibration import correct_black_level, load_black_levels
from src.raw_proc_estim import estimate_clipping, find_unclipped_median


class RawCalibrationTests(unittest.TestCase):
    def test_black_white_and_neighbouring_codes(self):
        raw = np.array([[0, 255, 256, 257, 4094, 4095]], dtype=np.uint16)
        original = raw.copy()
        corrected = correct_black_level(raw, 4095, np.full(4, 256), "SBGGR16")

        np.testing.assert_array_equal(corrected, [[0, 0, 0, 1, 4094, 4095]])
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(corrected.dtype, np.uint16)

    def test_each_bayer_channel_has_its_own_offset(self):
        levels = np.array([100, 200, 300, 400])  # R, Gr, Gb, B
        black_frames = {
            "RGGB": [[100, 200], [300, 400]],
            "BGGR": [[400, 300], [200, 100]],
            "GRBG": [[200, 100], [400, 300]],
            "GBRG": [[300, 400], [100, 200]],
        }
        for pattern, values in black_frames.items():
            with self.subTest(pattern=pattern):
                raw = np.array(values, dtype=np.uint16)
                corrected = correct_black_level(raw, 4095, levels, pattern)
                np.testing.assert_array_equal(corrected, np.zeros((2, 2)))
                raw[:] = 4095
                corrected = correct_black_level(raw, 4095, levels, pattern)
                np.testing.assert_array_equal(corrected, raw)

    def test_metadata_scale_and_frame_index(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            records = [
                {"frame_index": 1, "metadata": {"SensorBlackLevels": [8192] * 4}},
                {"frame_index": 0, "metadata": {"SensorBlackLevels": [4096] * 4}},
            ]
            metadata = directory / "metadata.jsonl"
            metadata.write_text("\n".join(json.dumps(r) for r in records))
            for bits, expected in ((10, [64, 128]), (12, [256, 512])):
                levels = load_black_levels(directory, bits, 2)
                np.testing.assert_array_equal(levels[:, 0], expected)
            with self.assertRaises(ValueError):
                load_black_levels(directory, 12, 3)
            metadata.write_text(json.dumps({"frame_index": 0, "metadata": {}}))
            with self.assertRaises(ValueError):
                load_black_levels(directory, 12, 1)

    def test_estimates_use_corrected_histogram(self):
        raw = np.array([[255, 256, 256, 512, 4095]], dtype=np.uint16)
        corrected = correct_black_level(raw, 4095, np.full(4, 256), "SBGGR16")
        histogram = np.bincount(corrected.ravel(), minlength=4096)
        self.assertEqual(histogram[0], 3)
        self.assertEqual(histogram[-1], 1)
        self.assertEqual(find_unclipped_median(histogram, 4095), 273)
        self.assertLess(estimate_clipping(histogram, 4095), 0)

    def test_dark_frame_is_not_normalized_by_its_minimum(self):
        raw = np.full((2, 2), 300, dtype=np.uint16)
        corrected = correct_black_level(raw, 4095, np.full(4, 256), "SBGGR16")
        np.testing.assert_array_equal(corrected, np.full((2, 2), 47))


if __name__ == "__main__":
    unittest.main()
