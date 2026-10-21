import numpy as np

def closed_form_mean_var(curr_recipient_count: np.ndarray, provider, l_edge_count) -> tuple[int, np.float64, np.float64]:
    # 샘플러가 r != src로 재추출하는 분포와 일치 시켜야함.
    p = curr_recipient_count.astype(np.float64)
    p[provider] = 0.0          
    p /= p.sum()

    n = l_edge_count
    h = (p ** 2).sum()
    t = (p ** 3).sum()
    closed_mean = n + n * (n - 1) * h
    closed_var = 2 * n * (n - 1) * h * (1 - h) + 4 * n * (n - 1) * (n - 2) * (t - h * h)
    return (n, closed_mean, closed_var)