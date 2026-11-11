"""Logistic regression in NumPy: median fill, scaling, Newton fit and Platt calibration.

Everything that is fitted here is plain NumPy: no scikit-learn estimator is
used. The steps follow the project's fit boundaries:

* ``NewtonLogisticRegression.fit`` on the train split: median fill values,
  scaling and coefficients.
* ``PlattCalibrator.fit`` on the calibration split, with the model frozen.
* threshold and alert policy on validation (``policy.tune_threshold``).

The model minimises the class-weighted log loss plus an L2 penalty on the
coefficients (not the intercept)::

    J(w, b) = sum_i s_i * logloss(y_i, sigmoid(x_i . w + b)) + (lam / 2) * ||w||^2

with ``s_i = n / (2 * n_class(y_i))`` ("balanced" weights) and ``lam = 1 / C``,
the same objective as scikit-learn's ``LogisticRegression(C=C,
class_weight="balanced")``. It is solved with Newton's method (iteratively
reweighted least squares), which converges in a few steps for this size.

This is the NumPy counterpart of the ``logistic_regression`` candidate in
``modeling.py`` (sklearn pipeline: median imputer, scaler, LogisticRegression
with C=0.3 and balanced weights). Both minimise the same objective, so their
coefficients agree; ``scripts/evaluate_logistic.py`` checks this. The median
fill here works on the feature matrix, unlike ``features.fit_training_fallback``,
which fills raw measurements before features are built.
"""

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-row weights n / (2 * n_class), so both classes carry equal total weight."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        raise ValueError("training labels need both classes")
    return (len(y) / (2.0 * counts))[y]


def weighted_log_loss(y, p, weights) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.sum(weights * (y * np.log(p) + (1 - y) * np.log(1 - p))))


@dataclass
class MedianScaler:
    """Training-split medians for missing values, then standardisation."""

    medians: np.ndarray = None
    means: np.ndarray = None
    scales: np.ndarray = None

    def fit(self, X: np.ndarray) -> "MedianScaler":
        X = np.asarray(X, dtype=np.float64)
        observed = ~np.isnan(X)
        # A column with no training values gets 0: it carries no information.
        with np.errstate(all="ignore"):
            medians = np.nanmedian(np.where(observed.any(axis=0), X, 0.0), axis=0)
        self.medians = np.where(observed.any(axis=0), medians, 0.0)
        filled = np.where(observed, X, self.medians)
        self.means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        self.scales = np.where(scales > 0, scales, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isnan(X), self.medians, X)
        return (filled - self.means) / self.scales


@dataclass
class NewtonLogisticRegression:
    """L2-regularised, class-weighted logistic regression fitted by Newton's method."""

    C: float = 0.3
    max_iter: int = 50
    tol: float = 1e-8
    coef_: np.ndarray = None
    intercept_: float = 0.0
    history: list = field(default_factory=list)
    converged: bool = False
    scaler: MedianScaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NewtonLogisticRegression":
        y = np.asarray(y, dtype=np.float64)
        self.scaler = MedianScaler().fit(X)
        Z = self.scaler.transform(X)
        n, d = Z.shape
        design = np.hstack([Z, np.ones((n, 1))])  # last column is the intercept
        weights = balanced_weights(y.astype(int))
        lam = 1.0 / self.C
        penalty = np.full(d + 1, lam)
        penalty[-1] = 0.0  # the intercept is not penalised
        theta = np.zeros(d + 1)
        self.history = []
        for _ in range(self.max_iter):
            p = sigmoid(design @ theta)
            loss = weighted_log_loss(y, p, weights) + 0.5 * lam * float(theta[:-1] @ theta[:-1])
            self.history.append(loss)
            gradient = design.T @ (weights * (p - y)) + penalty * theta
            curvature = weights * p * (1 - p)
            hessian = (design * curvature[:, None]).T @ design + np.diag(penalty)
            step = np.linalg.solve(hessian + 1e-10 * np.eye(d + 1), gradient)
            theta -= step
            # Newton decrement: a scale-free stopping rule.
            if float(gradient @ step) / 2 < self.tol:
                self.converged = True
                break
        p = sigmoid(design @ theta)
        self.history.append(weighted_log_loss(y, p, weights) + 0.5 * lam * float(theta[:-1] @ theta[:-1]))
        self.coef_, self.intercept_ = theta[:-1], float(theta[-1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])


@dataclass
class PlattCalibrator:
    """p = sigmoid(a * score + b), fitted by Newton's method on held-out scores."""

    a: float = 1.0
    b: float = 0.0
    max_iter: int = 100

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if len(np.unique(y)) != 2:
            raise ValueError("calibration labels need both classes")
        # Platt's smoothed targets avoid infinite weights on separable data.
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        target = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        design = np.column_stack([scores, np.ones_like(scores)])
        theta = np.array([0.0, np.log((n_pos + 1) / (n_neg + 1))])
        for _ in range(self.max_iter):
            p = sigmoid(design @ theta)
            gradient = design.T @ (p - target)
            hessian = (design * (p * (1 - p))[:, None]).T @ design + 1e-12 * np.eye(2)
            step = np.linalg.solve(hessian, gradient)
            theta -= step
            if np.abs(step).max() < 1e-10:
                break
        self.a, self.b = float(theta[0]), float(theta[1])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(scores, dtype=np.float64) + self.b)


@dataclass
class CalibratedLogistic:
    """Frozen logistic regression plus a Platt calibrator; sklearn-style predict_proba."""

    model: NewtonLogisticRegression
    calibrator: PlattCalibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.calibrator.transform(self.model.decision_function(X))
        return np.column_stack([1 - p, p])
