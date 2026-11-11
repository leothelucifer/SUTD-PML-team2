"""Hidden Markov model features: state probabilities that use only past and current hours."""

import numpy as np


def forward_filter(start: np.ndarray, transition: np.ndarray, emission: np.ndarray) -> np.ndarray:
    """Return P(state at hour t | observations up to hour t) for every hour.

    start:      shape (N,)   probability of each state at the first hour (pi)
    transition: shape (N, N) transition[i, j] = P(next state j | current state i) (A)
    emission:   shape (T, N) emission[t, i] = likelihood of hour t's observation in state i (b)
    """
    hours, states = emission.shape
    filtered = np.zeros((hours, states))
    predicted = start
    for t in range(hours):
        joint = predicted * emission[t]          # update with hour t's observation
        total = joint.sum()
        if total <= 0:
            raise ValueError(f"hour {t}: observation has zero likelihood in every state")
        filtered[t] = joint / total              # normalise so the row sums to 1
        predicted = filtered[t] @ transition     # predict the next hour
    return filtered


def gaussian_emissions(observations: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Return emission[t, i]: how well hour t's measurements fit state i.

    observations: shape (T, D) measured values, NaN where not measured
    means:        shape (N, D) mean of each measurement in each state
    stds:         shape (N, D) standard deviation of each measurement in each state

    Each measurement uses a normal (Gaussian) density, and the measurements of
    one hour are multiplied together. A missing measurement is skipped, so an
    hour with nothing measured gives every state the same value. Each hour is
    rescaled so its largest value is 1; forward_filter normalises every hour,
    so this does not change its result but stops very small numbers rounding to 0.
    """
    if np.any(stds <= 0):
        raise ValueError("standard deviations must be positive")
    x = observations[:, None, :]                       # (T, 1, D)
    z = (x - means[None, :, :]) / stds[None, :, :]     # (T, N, D)
    log_density = -0.5 * z**2 - np.log(stds)[None, :, :] - 0.5 * np.log(2 * np.pi)
    log_density = np.where(np.isnan(x), 0.0, log_density)   # skip missing measurements
    log_emission = log_density.sum(axis=2)             # multiply = add the logs
    log_emission -= log_emission.max(axis=1, keepdims=True)
    return np.exp(log_emission)