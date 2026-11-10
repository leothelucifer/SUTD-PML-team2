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