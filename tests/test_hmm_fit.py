import unittest

import numpy as np

from sepsis_warning.hmm import fit_hmm, forward_backward

START = np.array([0.8, 0.2])
TRANSITION = np.array([[0.9, 0.1], [0.3, 0.7]])
HIGH_HEART_RATE = np.array([0.2, 0.7])


def simulate(n_stays, hours, seed):
    """Two states: heart rate about 80 (sd 8) or about 110 (sd 10)."""
    rng = np.random.default_rng(seed)
    means, stds = np.array([80.0, 110.0]), np.array([8.0, 10.0])
    stays = []
    for _ in range(n_stays):
        state = rng.choice(2, p=START)
        rows = []
        for _ in range(hours):
            rows.append([rng.normal(means[state], stds[state])])
            state = rng.choice(2, p=TRANSITION[state])
        stays.append(np.array(rows))
    return stays


class BaumWelch(unittest.TestCase):
    def test_e_step_matches_hand_calculation(self):
        log_emission = np.log(np.array([HIGH_HEART_RATE, HIGH_HEART_RATE]))
        gamma, xi_sum, log_likelihood = forward_backward(START, TRANSITION, log_emission)
        np.testing.assert_allclose(gamma[0], [0.04 / 0.117, 0.077 / 0.117])
        np.testing.assert_allclose(xi_sum, np.array([[0.0288, 0.0112], [0.0084, 0.0686]]) / 0.117)
        self.assertAlmostEqual(log_likelihood, np.log(0.117))

    def test_log_likelihood_never_decreases(self):
        _, history = fit_hmm(simulate(30, 40, seed=0), n_states=2, tol=0)
        self.assertTrue(np.all(np.diff(history) >= -1e-6))

    def test_recovers_known_parameters(self):
        model, _ = fit_hmm(simulate(200, 50, seed=1), n_states=2)
        order = np.argsort(model.means[:, 0])
        np.testing.assert_allclose(model.means[order, 0], [80, 110], atol=2)
        np.testing.assert_allclose(model.stds[order, 0], [8, 10], atol=1.5)
        np.testing.assert_allclose(model.transition[np.ix_(order, order)], TRANSITION, atol=0.05)

    def test_missing_hours_are_allowed(self):
        stays = simulate(30, 40, seed=2)
        stays[0][5:10] = np.nan
        model, _ = fit_hmm(stays, n_states=2)
        probabilities = model.filter(stays[0])
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(40))

    def test_same_result_every_run(self):
        stays = simulate(20, 30, seed=3)
        first, _ = fit_hmm(stays, n_states=2)
        second, _ = fit_hmm(stays, n_states=2)
        np.testing.assert_array_equal(first.means, second.means)

    def test_filter_does_not_look_ahead(self):
        stays = simulate(30, 40, seed=4)
        model, _ = fit_hmm(stays, n_states=2)
        changed = stays[0].copy()
        changed[20:] = 200.0
        np.testing.assert_array_equal(model.filter(stays[0])[:20], model.filter(changed)[:20])


if __name__ == "__main__":
    unittest.main()
