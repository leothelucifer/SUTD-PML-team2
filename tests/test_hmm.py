import unittest

import numpy as np

from sepsis_warning.hmm import forward_filter

START = np.array([0.8, 0.2])
TRANSITION = np.array([[0.9, 0.1], [0.3, 0.7]])
HIGH_HEART_RATE = np.array([0.2, 0.7])


class ForwardFilter(unittest.TestCase):
    def test_matches_hand_calculation(self):
        emission = np.array([HIGH_HEART_RATE, HIGH_HEART_RATE])
        filtered = forward_filter(START, TRANSITION, emission)
        np.testing.assert_allclose(filtered[0], [0.16 / 0.30, 0.14 / 0.30])
        np.testing.assert_allclose(filtered[1], [0.124 / 0.39, 0.266 / 0.39])

    def test_each_hour_sums_to_one(self):
        emission = np.random.default_rng(0).uniform(0.01, 1, size=(20, 2))
        filtered = forward_filter(START, TRANSITION, emission)
        np.testing.assert_allclose(filtered.sum(axis=1), np.ones(20))

    def test_later_hours_do_not_change_earlier_ones(self):
        emission = np.random.default_rng(1).uniform(0.01, 1, size=(10, 2))
        changed = emission.copy()
        changed[6:] = np.random.default_rng(2).uniform(0.01, 1, size=(4, 2))
        before = forward_filter(START, TRANSITION, emission)
        after = forward_filter(START, TRANSITION, changed)
        np.testing.assert_array_equal(before[:6], after[:6])


if __name__ == "__main__":
    unittest.main()