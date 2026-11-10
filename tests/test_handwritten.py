"""Checks of the hand-written NumPy model against its definition and scikit-learn."""

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from sepsis_warning.handwritten import (
    CalibratedNumpyModel, NumpyLogisticRegression, PlattCalibrator, Preprocessor, balanced_weights, sigmoid,
)


def make_data(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * np.arange(1, d + 1)
    logits = X @ np.linspace(1, -1, d) / 3 - 2.0
    y = (rng.random(n) < sigmoid(logits)).astype(int)
    X[rng.random((n, d)) < 0.1] = np.nan
    return X, y


class HandwrittenModel(unittest.TestCase):
    def test_sigmoid_is_stable_at_extremes(self):
        p = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(p, [0.0, 0.5, 1.0])
        self.assertFalse(np.isnan(p).any())

    def test_balanced_weights_give_each_class_equal_total(self):
        y = np.array([0] * 90 + [1] * 10)
        w = balanced_weights(y)
        self.assertAlmostEqual(w[y == 0].sum(), w[y == 1].sum())
        self.assertAlmostEqual(w.sum(), len(y))
        with self.assertRaises(ValueError):
            balanced_weights(np.zeros(5, dtype=int))

    def test_preprocessor_uses_training_values_only(self):
        train = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
        prep = Preprocessor().fit(train)
        self.assertEqual(prep.medians[0], 2.0)
        self.assertEqual(prep.medians[1], 0.0)  # never observed in training
        other = np.array([[100.0, 5.0]])
        before = (prep.medians.copy(), prep.means.copy(), prep.scales.copy())
        prep.transform(other)
        for kept, now in zip(before, (prep.medians, prep.means, prep.scales), strict=True):
            np.testing.assert_array_equal(kept, now)

    def test_matches_sklearn_on_the_same_objective(self):
        X, y = make_data()
        model = NumpyLogisticRegression(C=0.3).fit(X, y)
        self.assertTrue(model.converged)
        reference = LogisticRegression(C=0.3, class_weight="balanced", solver="lbfgs", max_iter=5000, tol=1e-12)
        reference.fit(model.preprocessor.transform(X), y)
        np.testing.assert_allclose(model.coef_, reference.coef_[0], atol=1e-5)
        self.assertAlmostEqual(model.intercept_, reference.intercept_[0], places=5)

    def test_newton_objective_never_increases(self):
        X, y = make_data(seed=1)
        model = NumpyLogisticRegression(C=1.0).fit(X, y)
        self.assertTrue(np.all(np.diff(model.history) <= 1e-9))

    def test_stronger_penalty_shrinks_coefficients(self):
        X, y = make_data(seed=2)
        loose = NumpyLogisticRegression(C=10.0).fit(X, y)
        tight = NumpyLogisticRegression(C=0.01).fit(X, y)
        self.assertLess(np.linalg.norm(tight.coef_), np.linalg.norm(loose.coef_))

    def test_platt_recovers_known_slope_and_offset(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(size=20000)
        y = (rng.random(20000) < sigmoid(1.5 * scores - 1.0)).astype(int)
        cal = PlattCalibrator().fit(scores, y)
        self.assertAlmostEqual(cal.a, 1.5, delta=0.1)
        self.assertAlmostEqual(cal.b, -1.0, delta=0.1)

    def test_calibrated_model_returns_valid_probabilities(self):
        X, y = make_data(seed=4)
        model = NumpyLogisticRegression().fit(X[:400], y[:400])
        cal = PlattCalibrator().fit(model.decision_function(X[400:500]), y[400:500])
        p = CalibratedNumpyModel(model, cal).predict_proba(X[500:])
        self.assertEqual(p.shape, (100, 2))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
