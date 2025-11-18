import numpy as np

MIN_SIGMA, MAX_SIGMA, N = 0.001, 2, 100


def get_sigma_scheduler(
    min_sigma: float = None, max_sigma: float = None, n: int = None
) -> np.ndarray:

    if min_sigma is None:
        min_sigma = MIN_SIGMA
    if max_sigma is None:
        max_sigma = MAX_SIGMA
    if n is None:
        n = N
    sigmas = 10 ** np.linspace(np.log10(MIN_SIGMA), np.log10(MAX_SIGMA), n)
    return sigmas
