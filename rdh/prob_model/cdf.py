import numpy as np

def build_cdf_r(recipient_count: np.ndarray) -> np.ndarray:
    raw_p_r = np.cumsum(recipient_count).astype(np.float64)
    return (raw_p_r / raw_p_r[-1])

def sample_from_cdf_r(src: np.ndarray, cdf: np.ndarray, size, rng) -> np.ndarray:
    null_sample = np.searchsorted(cdf, rng.uniform(size=size), side='right')

    retry = 0
    loop = src == null_sample
    new_sample_needed = np.sum(loop)
    while new_sample_needed > 0:
        null_sample[loop] = np.searchsorted(cdf, rng.uniform(size=(new_sample_needed, )), side='right')

        loop = src == null_sample
        new_sample_needed = np.sum(loop)
        retry += 1
    #print(f"retry : {retry}")

    assert (null_sample != src).all()
    return null_sample